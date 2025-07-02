import os
import re
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import fitz 
import ollama  
from PIL import Image, ImageOps, ImageChops, UnidentifiedImageError  
import io
import streamlit as st
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse , unquote, urljoin
import logging
import arxiv
import time
import pandas as pd
import markdown
from difflib import SequenceMatcher

# ----- Mistral OCR imports -----
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse

# ----- LangChain and related RAG imports -----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----- Multi-Agent imports -----
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# --------------------
# 🔧 CONSTANTS & PATHS
# --------------------
PDFS_DIRECTORY = "pdfs/"
REFERENCES_DIRECTORY = "references/"
SAVE_DIRECTORY = r"C:\Users\Kevin\Desktop\master\RNN&Transformer\code\code\VB"
AGENT_LOGS_DIR = os.path.join(SAVE_DIRECTORY, "agent_logs")

os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs(REFERENCES_DIRECTORY, exist_ok=True)
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
os.makedirs(AGENT_LOGS_DIR, exist_ok=True)
IMG_DIR = os.path.join(SAVE_DIRECTORY, "extracted_images")
os.makedirs(IMG_DIR, exist_ok=True)

# --------------------
# 🔑 EXTERNAL SERVICES
# --------------------
api_key = "LzMuOCLKNkbir3yph5KFTjMjdatT3ykX"  # TODO: move to env var
client = Mistral(api_key=api_key)

# LangChain LLM & Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
def get_vector_store() -> InMemoryVectorStore:
    """Globally unique vector store, stored in session_state to avoid rebuilding on every rerun."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = InMemoryVectorStore(embeddings)
    return st.session_state.vector_store
llm_model = OllamaLLM(model="gemma3:latest", temperature=0.7)
vision_model = OllamaLLM(model="gemma3:latest")

# --------------------
# 🧰 HELPER FUNCTIONS
# --------------------

def ocr_pdf(pdf_path: str) -> OCRResponse:
    """Run Mistral OCR over a single PDF and return the full response."""
    pdf_file = Path(pdf_path)
    uploaded = client.files.upload(
        file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    return client.ocr.process(
        document=DocumentURLChunk(document_url=signed.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )


def extract_images_with_pymupdf(pdf_path: str,
                                auto_crop: bool = True,
                                caption_height: float = 40.0) -> List[Dict]:
    sidebar = st.sidebar
    sidebar.markdown("## 📸 PyMuPDF Extracted")

    doc, results = fitz.open(pdf_path), []
    have_ollama = _check_ollama(sidebar)
    figure_idx = 1
    for page_i in range(len(doc)):
        page = doc.load_page(page_i)
        text_blocks = page.get_text("blocks")

        for img_i, info in enumerate(page.get_images(full=True)):
            xref = info[0]

            img_dict = doc.extract_image(xref)
            img_bytes = img_dict["image"]

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            bbox = rects[0]

            try:
                im = Image.open(io.BytesIO(img_bytes)); im.load()
            except UnidentifiedImageError:
                continue

            if auto_crop:
                bg = Image.new(im.mode, im.size, (255,)*len(im.getbands()))
                diff = ImageOps.invert(ImageChops.difference(im, bg).convert("L"))
                bb = diff.getbbox()
                if bb:
                    im = im.crop(bb)

            fname = f"figure{figure_idx:03d}.png"
            fpath = os.path.join(IMG_DIR, fname)
            im.save(fpath, "PNG")
            sidebar.image(im, caption=f"Figure {figure_idx}", width=120)

            raw_caption = _grab_caption(text_blocks, bbox, caption_height)

            llm_caption = ""
            if have_ollama:
                try:
                    abs_path = os.path.abspath(fpath)
                    prompt_txt = (
                        "📝  Examine the figure carefully and write 4-6 crisp sentences that explain:\n"
                        "1. What type of figure this is (e.g., bar chart, line graph, photo, model architecture, output example, confusion matrix, etc.).\n"
                        "2. What visual elements are present — describe key parts such as axes, labels, text, color maps, legends, nodes, or bounding boxes using the exact labels shown.\n"
                        "3. What the figure mainly shows or demonstrates — describe the key result, comparison, structure, or pattern that the image conveys.\n"
                        "4. Any numbers, thresholds, or units that stand out, or other noticeable visual markers.\n"
                        "5. If multiple elements are compared or presented, point out the most prominent difference (e.g., best/worst, clearest, most unusual, etc.).\n"
                        "6. (Optional) If a caption or task name appears inside or below the figure, quote it briefly.\n"
                        "Focus on what is visible inside the image; omit wider paper context unless it is printed in the figure. "
                        "Keep the answer ≤ 150 words and write in complete English sentences."
                    )

                    resp = ollama.chat(
                        model="qwen2.5vl:latest",
                        messages=[{
                            "role": "user",
                            "content": prompt_txt,
                            "images": [abs_path]
                        }],
                    )
                    llm_caption = resp["message"]["content"].strip()

                except Exception as e:
                    sidebar.warning(f"Vision description failed: {e}")

            results.append({
                "page": page_i,
                "figure": figure_idx,
                "filepath": fpath,
                "caption": f"{raw_caption}  {llm_caption}".strip()
            })
            figure_idx += 1

    doc.close()
    sidebar.success(f"✅ PyMuPDF exported {len(results)} PNG images in total")
    return results


def _check_ollama(sidebar) -> bool:
    try:
        _ = ollama.list()
        return True
    except Exception:
        sidebar.warning("⚠️ Local Ollama not found; Vision description will be skipped")
        return False


def _grab_caption(blocks, bbox, cap_h):
    y0, y1 = bbox.y1, bbox.y1 + cap_h
    lines = [
        (ty0, txt.strip())
        for (tx0, ty0, tx1, ty1, txt, *_) in blocks
        if txt.strip() and (tx1 > bbox.x0 and tx0 < bbox.x1) and (y0 < ty0 < y1)
    ]
    lines.sort(key=lambda t: t[0])
    return " ".join(t for _, t in lines)


def get_combined_text(ocr_resp: OCRResponse) -> str:
    """Concatenate markdown from every OCR-extracted page."""
    return "\n\n".join(page.markdown for page in ocr_resp.pages)

def create_documents_with_metadata(text: str, source: str) -> List[Document]:
    """Split raw text into overlapping chunks and wrap them in LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_text(text)
    now = datetime.now().isoformat()

    documents = []
    for i, chunk in enumerate(chunks):
        contains_fig = "[Figure " in chunk
        metadata = {
            "source": source,
            "chunk_id": i,
            "timestamp": now,
            "contains_figure": contains_fig
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

    return documents


def index_documents(documents: List[Document]) -> int:
    vs = get_vector_store()
    vs.add_documents(documents)
    return len(vs.store)


def retrieve_relevant_docs(query: str, k: int = 5) -> List[Document]:
    return get_vector_store().similarity_search(query, k=k)


def answer_question_with_rag(question: str) -> Dict[str, Any]:
    """Retrieve-and-generate: pull top K docs then feed to the LLM."""
    relevant_docs = retrieve_relevant_docs(question)
    if not relevant_docs:
        st.sidebar.write(f"🔍 Indexed docs: {len(get_vector_store().store)}")
        return {
            "answer": "Sorry, I couldn't find relevant information in the document to answer this question.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        [f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a professional document assistant. Please answer questions based on the provided context.

Context Information:
{context}

Question: {question}

Please note:
1. Only use information from the context to answer
2. If the context doesn't contain relevant information, state this clearly
3. Be accurate, specific, and well-organized in your response
4. Quote specific sources when possible

Answer:"""
    )
    chain = prompt | llm_model
    raw_answer = chain.invoke({"question": question, "context": context})
    answer_text = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)
    return {"answer": answer_text, "sources": relevant_docs, "context": context}


def generate_summary(full_text: str) -> str:
    """Summarise large documents by sampling beginning/middle/end if necessary."""
    if len(full_text) > 3000:
        parts = [
            full_text[:1000],
            full_text[len(full_text) // 2 - 500 : len(full_text) // 2 + 500],
            full_text[-1000:],
        ]
        context = "\n\n...\n\n".join(parts)
    else:
        context = full_text

    prompt = ChatPromptTemplate.from_template(
        """Please provide a comprehensive summary of the following document:

Document Content:
{context}

Include:
1. Main topics and key points
2. Important findings or conclusions
3. Key methodologies or techniques
4. Any significant data or results

Summary:"""
    )
    chain = prompt | llm_model
    return chain.invoke({"context": context})


def extract_references(full_text: str) -> List[str]:
    """Extract complete reference titles from the document text using multiple strategies."""
    references = []
    
    # Look for References/Bibliography section
    ref_patterns = [
        r'(?i)(References|Bibliography|Works Cited|Literature Cited)\s*\n+(.*?)(?=\n\n[A-Z]|\n\n\d+\.|\Z)',
        r'(?i)\n(References|Bibliography)\s*\n+(.*?)$',  # End of document
    ]
    
    ref_section = None
    for pattern in ref_patterns:
        ref_match = re.search(pattern, full_text, re.DOTALL)
        if ref_match:
            ref_section = ref_match.group(2)
            break
    
    if not ref_section:
        return []
    
    # Split into individual references
    ref_items = []
    lines = ref_section.split('\n')
    current_ref = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new reference
        if (re.match(r'^\[?\d+\]?\.?\s*[A-Z]', line) or  # [1] Author or 1. Author
            re.match(r'^\d+\)\s*[A-Z]', line) or          # 1) Author
            (len(current_ref) > 100 and line[0].isupper())):  # New line with capital
            if current_ref:
                ref_items.append(current_ref.strip())
            current_ref = line
        else:
            current_ref += " " + line
    
    if current_ref:
        ref_items.append(current_ref.strip())
    
    # Extract paper titles from each reference
    paper_titles = []
    for ref in ref_items:
        # Remove citation number
        ref = re.sub(r'^\[?\d+\]?\.?\s*', '', ref)
        ref = re.sub(r'^\d+\)\s*', '', ref)
        
        # Strategy 1: Look for quoted titles
        title_match = re.search(r'"([^"]+)"', ref)
        if title_match:
            paper_titles.append(title_match.group(1))
            continue
            
        # Strategy 2: Look for title in italics markers (if any)
        title_match = re.search(r'[_*]([^_*]+)[_*]', ref)
        if title_match:
            paper_titles.append(title_match.group(1))
            continue
        
        # Strategy 3: Parse common academic citation formats
        # Most formats: Authors (Year). Title. Journal/Conference.
        year_split = re.split(r'\(\d{4}\)\.?\s*', ref, maxsplit=1)
        if len(year_split) > 1:
            after_year = year_split[1]
            # Title is usually the first sentence after year
            sentences = re.split(r'\.\s+', after_year)
            if sentences:
                title = sentences[0].strip()
                title = re.sub(r'\.$', '', title)
                if len(title) > 10:  # Reasonable title length
                    paper_titles.append(title)
                    continue
        
        # Strategy 4: If no year, look for pattern: Authors. Title. Rest.
        parts = ref.split('. ')
        if len(parts) >= 2:
            # Skip author part (usually has commas and initials)
            for i, part in enumerate(parts[1:], 1):
                # Title characteristics: longer, no author patterns
                if (len(part) > 20 and 
                    not re.search(r'[A-Z]\.\s*[A-Z]', part) and  # No initials
                    not re.search(r',\s*[A-Z]\.', part)):  # No ", A."
                    paper_titles.append(part.strip())
                    break
    
    # Clean up titles
    cleaned_titles = []
    for title in paper_titles:
        # Remove extra whitespace
        title = ' '.join(title.split())
        # Remove trailing punctuation
        title = re.sub(r'[.,;:]+$', '', title)
        # Only keep if reasonable length
        if 10 < len(title) < 300:
            cleaned_titles.append(title)
    
    return cleaned_titles[:100]  # Return up to 30 references


# --------------------
# 📊 REPORT GENERATION FUNCTIONS
# --------------------

def generate_comprehensive_report(
    full_text: str, 
    summary: str, 
    agent_results_df: pd.DataFrame,
    pdf_name: str
) -> str:
    """Generate a comprehensive report combining all analysis results."""
    
    # Extract key sections from the full text
    sections = extract_paper_sections(full_text)
    
    # Generate the report
    report = f"""# 📊 Comprehensive Research Paper Analysis Report

**Document:** {pdf_name}  
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Executive Summary

{summary}

---

## 📑 Paper Structure Analysis

### Abstract
{sections.get('abstract', 'Abstract not found in the document.')}

### Introduction & Background
{sections.get('introduction', 'Introduction section not clearly identified.')}

### Methodology
{sections.get('methodology', 'Methodology section not clearly identified.')}

### Results & Findings
{sections.get('results', 'Results section not clearly identified.')}

### Conclusions
{sections.get('conclusions', 'Conclusions section not clearly identified.')}

---

## 📚 Reference Analysis

### Summary Statistics
- **Total References Extracted:** {len(agent_results_df) if agent_results_df is not None else 0}
- **Successfully Downloaded:** {len(agent_results_df[agent_results_df['Source'] == 'Downloaded PDF']) if agent_results_df is not None else 0}
- **Title-based Summaries:** {len(agent_results_df[agent_results_df['Source'] == 'Title-based (No PDF)']) if agent_results_df is not None else 0}

### Detailed Reference Summaries

"""

    # Add reference summaries
    if agent_results_df is not None and not agent_results_df.empty:
        for idx, row in agent_results_df.iterrows():
            report += f"""
#### {idx + 1}. {row['Paper Title']}
**Source:** {row['Source']}  
**Summary:** {row['Summary']}

---
"""
    else:
        report += "\nNo reference summaries available.\n\n"

    # Add key insights section
    report += """
## 💡 Key Insights and Connections

Based on the analysis of the main paper and its references, here are the key insights:

"""
    
    # Generate insights using LLM
    insights = generate_insights(full_text, agent_results_df)
    report += insights
    
    # Add recommendations
    report += """

## 🎯 Recommendations for Further Research

1. **Priority References to Read**: Based on the summaries, the most relevant references appear to be those focusing on similar methodologies or addressing related research questions.

2. **Gap Analysis**: Consider exploring areas that are mentioned but not thoroughly covered in the referenced works.

3. **Methodological Improvements**: Look for techniques used in referenced papers that could enhance the current research approach.

4. **Future Directions**: The combination of this paper and its references suggests several promising research directions.

---

## 📊 Appendix: Full Text Statistics

"""
    
    # Add text statistics
    stats = generate_text_statistics(full_text)
    report += stats
    
    return report


def extract_paper_sections(full_text: str) -> Dict[str, str]:
    """Extract major sections from the paper text."""
    sections = {}
    
    # Common section patterns
    section_patterns = {
        'abstract': r'(?i)(abstract|summary)\s*\n+(.*?)(?=\n\s*\n[A-Z]|\n\s*\n\d+\.)',
        'introduction': r'(?i)(introduction|background)\s*\n+(.*?)(?=\n\s*\n[A-Z]|\n\s*\n\d+\.)',
        'methodology': r'(?i)(method|methodology|materials and methods|experimental setup)\s*\n+(.*?)(?=\n\s*\n[A-Z]|\n\s*\n\d+\.)',
        'results': r'(?i)(results|findings|experiments)\s*\n+(.*?)(?=\n\s*\n[A-Z]|\n\s*\n\d+\.)',
        'conclusions': r'(?i)(conclusion|conclusions|summary and conclusions)\s*\n+(.*?)(?=\n\s*\n[A-Z]|\n\s*\nReferences)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            content = match.group(2).strip()
            # Limit to first 500 words
            words = content.split()
            if len(words) > 500:
                content = ' '.join(words[:500]) + "..."
            sections[section_name] = content
    
    return sections


def generate_insights(full_text: str, agent_results_df: pd.DataFrame) -> str:
    """Generate insights by analyzing the paper and its references."""
    
    # Prepare context
    ref_summaries = ""
    if agent_results_df is not None and not agent_results_df.empty:
        ref_summaries = "\n".join([
            f"- {row['Paper Title']}: {row['Summary']}" 
            for _, row in agent_results_df.head(5).iterrows()
        ])
    
    prompt = ChatPromptTemplate.from_template(
        """Based on the research paper and its reference summaries, provide 3-4 key insights about:

1. How this research connects to and builds upon previous work
2. The unique contributions of this paper
3. Emerging trends or patterns in the field
4. Potential research gaps or opportunities

Paper excerpt:
{paper_excerpt}

Top reference summaries:
{ref_summaries}

Provide insights in bullet points:"""
    )
    
    # Get first 1000 words of paper
    paper_excerpt = ' '.join(full_text.split()[:1000])
    
    chain = prompt | llm_model
    result = chain.invoke({
        "paper_excerpt": paper_excerpt,
        "ref_summaries": ref_summaries
    })
    
    return result.content if hasattr(result, "content") else str(result)


def generate_text_statistics(full_text: str) -> str:
    """Generate statistics about the document."""
    
    words = full_text.split()
    sentences = re.split(r'[.!?]+', full_text)
    
    # Count figures and tables mentioned
    figures = len(re.findall(r'(?i)figure\s*\d+', full_text))
    tables = len(re.findall(r'(?i)table\s*\d+', full_text))
    
    stats = f"""
- **Total Words:** {len(words):,}
- **Total Sentences:** {len(sentences):,}
- **Average Words per Sentence:** {len(words) / max(len(sentences), 1):.1f}
- **Figures Mentioned:** {figures}
- **Tables Mentioned:** {tables}
- **Unique Words:** {len(set(word.lower() for word in words)):,}
"""
    
    return stats


def save_report(report: str, pdf_name: str) -> str:
    """Save the comprehensive report to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdf_name.replace('.pdf', '')
    
    report_filename = f"report_{base_name}_{timestamp}.md"
    report_filepath = os.path.join(SAVE_DIRECTORY, report_filename)
    
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_filepath


# --------------------
# 💾 SAVE FUNCTIONS
# --------------------

def save_ocr_result(ocr_text: str, pdf_name: str, ocr_response: OCRResponse = None):
    """儲存OCR處理後的文字結果到指定路徑"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdf_name.replace('.pdf', '')
    
    txt_filename = f"ocr_{base_name}_{timestamp}.txt"
    txt_filepath = os.path.join(SAVE_DIRECTORY, txt_filename)
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"OCR Result for: {pdf_name}\n")
        f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(ocr_text)
    
    md_filename = f"ocr_{base_name}_{timestamp}.md"
    md_filepath = os.path.join(SAVE_DIRECTORY, md_filename)
    
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(f"# OCR Result for: {pdf_name}\n\n")
        f.write(f"**Processed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(ocr_text)
    
    if ocr_response:
        json_filename = f"ocr_detailed_{base_name}_{timestamp}.json"
        json_filepath = os.path.join(SAVE_DIRECTORY, json_filename)
        
        detailed_data = {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": pdf_name,
            "total_pages": len(ocr_response.pages),
            "pages": []
        }
        
        for i, page in enumerate(ocr_response.pages):
            page_data = {
                "page_number": i + 1,
                "markdown": page.markdown,
                "has_image": hasattr(page, 'image_base64') and page.image_base64 is not None
            }
            detailed_data["pages"].append(page_data)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    
    return txt_filepath, md_filepath


# --------------------
# 🤖 MULTI-AGENT SYSTEM
# --------------------

class AgentCommunicationHub:
    """Central hub for agent communication and result storage"""
    def __init__(self):
        self.messages = []
        self.results = {}
        self.failed_downloads = []
    
    def log_message(self, sender: str, receiver: str, message: str):
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "message": message
        })
    
    def store_result(self, paper_title: str, summary: str, source: str):
        self.results[paper_title] = {
            "summary": summary,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_failed_download(self, paper_title: str):
        self.failed_downloads.append(paper_title)
    
    def get_results_df(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=["Paper Title", "Summary", "Source"])
        
        data = []
        for title, info in self.results.items():
            data.append({
                "Paper Title": title,
                "Summary": info["summary"],
                "Source": info["source"]
            })
        return pd.DataFrame(data)


# Initialize communication hub in session state
def get_comm_hub() -> AgentCommunicationHub:
    if "comm_hub" not in st.session_state:
        st.session_state.comm_hub = AgentCommunicationHub()
    return st.session_state.comm_hub


# Child Agent: Web Agent with multiple sources
def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings with better normalization"""
    # More aggressive normalization
    a_norm = re.sub(r'[^\w\s]', '', a.lower()).strip()
    b_norm = re.sub(r'[^\w\s]', '', b.lower()).strip()
    
    # Remove common academic words that might cause false negatives
    common_words = {'paper', 'study', 'analysis', 'research', 'investigation', 'approach', 'method'}
    a_words = [w for w in a_norm.split() if w not in common_words]
    b_words = [w for w in b_norm.split() if w not in common_words]
    
    a_clean = ' '.join(a_words)
    b_clean = ' '.join(b_words)
    
    return SequenceMatcher(None, a_clean, b_clean).ratio()

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from search results text more reliably"""
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    return list(set(urls))  # Remove duplicates

def is_pdf_url(url: str) -> bool:
    """Check if URL likely points to a PDF"""
    url_lower = url.lower()
    return (url_lower.endswith('.pdf') or 
            'pdf' in url_lower or 
            'download' in url_lower or
            'arxiv.org/pdf' in url_lower)

def clean_paper_title(title: str) -> str:
    """Clean paper title for various uses"""
    # Remove special characters but keep important ones
    cleaned = re.sub(r'[^\w\s\-:()]', '', title)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def generate_search_variations(paper_title: str) -> List[str]:
    """Generate different search query variations"""
    variations = []
    
    # Original title
    variations.append(f'"{paper_title}"')
    
    # Without quotes
    variations.append(paper_title)
    
    # Key terms only (first 5 words)
    words = paper_title.split()[:5]
    variations.append(' '.join(words))
    
    # Remove common academic words
    common_words = {'a', 'an', 'the', 'on', 'for', 'and', 'or', 'but', 'in', 'with', 'by'}
    important_words = [w for w in paper_title.split() if w.lower() not in common_words]
    if len(important_words) >= 3:
        variations.append(' '.join(important_words[:4]))
    
    return variations

def download_pdf_with_retries(url: str, max_retries: int = 3) -> Optional[bytes]:
    """Download PDF with retries and better error handling"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or 'octet-stream' in content_type:
                return response.content
            
            # Sometimes PDFs are served without proper content-type
            if response.content.startswith(b'%PDF'):
                return response.content
                
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

def search_semantic_scholar(paper_title: str) -> List[str]:
    """Search Semantic Scholar API for papers"""
    try:
        # Semantic Scholar API endpoint
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': paper_title,
            'limit': 5,
            'fields': 'title,url,openAccessPdf'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pdf_urls = []
            
            for paper in data.get('data', []):
                if similarity_score(paper.get('title', ''), paper_title) > 0.7:
                    # Check for open access PDF
                    open_access = paper.get('openAccessPdf')
                    if open_access and open_access.get('url'):
                        pdf_urls.append(open_access['url'])
            
            return pdf_urls
    except Exception as e:
        logging.warning(f"Semantic Scholar search failed: {str(e)}")
    
    return []

def web_agent_func(paper_title: str) -> str:
    """Enhanced web agent with multiple strategies and better error handling"""
    comm_hub = get_comm_hub()
    
    # Clean paper title for filename
    clean_title = re.sub(r'[^\w\s-]', '', paper_title)
    clean_title = re.sub(r'[-\s]+', '_', clean_title)
    
    comm_hub.log_message("web_agent", "system", f"Starting enhanced search for: {paper_title}")
    
    # Strategy 1: arXiv search (most reliable)
    try:
        comm_hub.log_message("web_agent", "system", "Trying arXiv search...")
        
        # Try multiple arXiv search strategies
        search_strategies = [
            f'ti:"{paper_title}"',  # Exact title
            f'all:"{paper_title}"',  # All fields
            ' '.join(paper_title.split()[:5])  # Key terms
        ]
        
        for strategy in search_strategies:
            search = arxiv.Search(
                query=strategy,
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                similarity = similarity_score(paper.title, paper_title)
                comm_hub.log_message("web_agent", "system", 
                    f"arXiv match: '{paper.title}' (similarity: {similarity:.2f})")
                
                if similarity > 0.75:  # Lowered threshold
                    try:
                        filename = f"{clean_title}_arxiv.pdf"
                        path = os.path.join(REFERENCES_DIRECTORY, filename)
                        paper.download_pdf(dirpath=REFERENCES_DIRECTORY, filename=filename)
                        comm_hub.log_message("web_agent", "mother_agent", f"Downloaded from arXiv: {filename}")
                        return f"✅ Downloaded '{paper_title}' from arXiv to {filename}"
                    except Exception as e:
                        comm_hub.log_message("web_agent", "system", f"arXiv download failed: {str(e)}")
                        
    except Exception as e:
        comm_hub.log_message("web_agent", "system", f"arXiv search failed: {str(e)}")
    
    # Strategy 2: Semantic Scholar
    try:
        comm_hub.log_message("web_agent", "system", "Trying Semantic Scholar...")
        pdf_urls = search_semantic_scholar(paper_title)
        
        for pdf_url in pdf_urls:
            comm_hub.log_message("web_agent", "system", f"Trying Semantic Scholar PDF: {pdf_url}")
            pdf_content = download_pdf_with_retries(pdf_url)
            
            if pdf_content:
                filename = f"{clean_title}_semantic.pdf"
                path = os.path.join(REFERENCES_DIRECTORY, filename)
                with open(path, "wb") as f:
                    f.write(pdf_content)
                comm_hub.log_message("web_agent", "mother_agent", f"Downloaded from Semantic Scholar: {filename}")
                return f"✅ Downloaded '{paper_title}' from Semantic Scholar to {filename}"
                
    except Exception as e:
        comm_hub.log_message("web_agent", "system", f"Semantic Scholar search failed: {str(e)}")
    
    # Strategy 3: Enhanced DuckDuckGo search
    try:
        from duckduckgo_search import DDGS
        
        comm_hub.log_message("web_agent", "system", "Trying enhanced DuckDuckGo search...")
        
        search_variations = generate_search_variations(paper_title)
        
        for variation in search_variations:
            try:
                # Use the newer duckduckgo-search library if available
                with DDGS() as ddgs:
                    # Search for PDFs
                    pdf_query = f"{variation} filetype:pdf"
                    results = list(ddgs.text(pdf_query, max_results=10))
                    
                    for result in results:
                        url = result.get('href', '')
                        title = result.get('title', '')
                        
                        # Check if this looks like our paper
                        if similarity_score(title, paper_title) > 0.6 and is_pdf_url(url):
                            comm_hub.log_message("web_agent", "system", f"Trying DuckDuckGo PDF: {url}")
                            pdf_content = download_pdf_with_retries(url)
                            
                            if pdf_content:
                                filename = f"{clean_title}_web.pdf"
                                path = os.path.join(REFERENCES_DIRECTORY, filename)
                                with open(path, "wb") as f:
                                    f.write(pdf_content)
                                comm_hub.log_message("web_agent", "mother_agent", f"Downloaded from web: {filename}")
                                return f"✅ Downloaded '{paper_title}' from web to {filename}"
                        
            except Exception as e:
                comm_hub.log_message("web_agent", "system", f"DuckDuckGo variation '{variation}' failed: {str(e)}")
                continue
                
    except ImportError:
        # Fallback to original DuckDuckGo search if new library not available
        comm_hub.log_message("web_agent", "system", "Using fallback DuckDuckGo search...")
        try:
            search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=15)
            search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
            
            for variation in generate_search_variations(paper_title):
                search_query = f'{variation} filetype:pdf'
                search_results = search_tool.run(search_query)
                
                # Extract URLs more reliably
                urls = extract_urls_from_text(search_results)
                pdf_urls = [url for url in urls if is_pdf_url(url)]
                
                for pdf_url in pdf_urls[:5]:  # Limit attempts
                    comm_hub.log_message("web_agent", "system", f"Trying extracted PDF: {pdf_url}")
                    pdf_content = download_pdf_with_retries(pdf_url)
                    
                    if pdf_content:
                        filename = f"{clean_title}_web.pdf"
                        path = os.path.join(REFERENCES_DIRECTORY, filename)
                        with open(path, "wb") as f:
                            f.write(pdf_content)
                        comm_hub.log_message("web_agent", "mother_agent", f"Downloaded from web: {filename}")
                        return f"✅ Downloaded '{paper_title}' from web to {filename}"
        except Exception as e:
            comm_hub.log_message("web_agent", "system", f"Fallback DuckDuckGo search failed: {str(e)}")
    
    # Strategy 4: Try direct Google Scholar scraping (as last resort)
    try:
        comm_hub.log_message("web_agent", "system", "Trying Google Scholar search...")
        
        # This is a simplified approach - you might want to use scholarly library
        scholar_query = f"https://scholar.google.com/scholar?q={requests.utils.quote(paper_title)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(scholar_query, headers=headers, timeout=10)
        if response.status_code == 200:
            # Look for PDF links in the response
            pdf_links = re.findall(r'href="([^"]*\.pdf[^"]*)"', response.text)
            
            for pdf_link in pdf_links[:3]:  # Try first 3 PDF links
                if not pdf_link.startswith('http'):
                    pdf_link = urljoin('https://scholar.google.com', pdf_link)
                
                comm_hub.log_message("web_agent", "system", f"Trying Google Scholar PDF: {pdf_link}")
                pdf_content = download_pdf_with_retries(pdf_link)
                
                if pdf_content:
                    filename = f"{clean_title}_scholar.pdf"
                    path = os.path.join(REFERENCES_DIRECTORY, filename)
                    with open(path, "wb") as f:
                        f.write(pdf_content)
                    comm_hub.log_message("web_agent", "mother_agent", f"Downloaded from Google Scholar: {filename}")
                    return f"✅ Downloaded '{paper_title}' from Google Scholar to {filename}"
        
    except Exception as e:
        comm_hub.log_message("web_agent", "system", f"Google Scholar search failed: {str(e)}")
    
    # If all strategies fail
    comm_hub.log_message("web_agent", "mother_agent", f"All download strategies failed for: {paper_title}")
    comm_hub.add_failed_download(paper_title)
    return f"❌ Could not download '{paper_title}' from any source after trying multiple strategies"

# Child Agent: Code Agent
def code_agent_func(task: str) -> str:
    """Generate code for downloading papers"""
    comm_hub = get_comm_hub()
    comm_hub.log_message("code_agent", "system", f"Generating code for: {task}")
    
    prompt = ChatPromptTemplate.from_template(
        """Generate Python code to download a research paper with the following requirements:
        
Task: {task}

The code should:
1. Try multiple sources (direct URL, arXiv, semantic scholar)
2. Handle errors gracefully
3. Save to a specified directory
4. Return the file path if successful

Please provide complete, runnable code with all necessary imports.

Code:"""
    )
    
    chain = prompt | llm_model
    result = chain.invoke({"task": task})
    code = result.content if hasattr(result, "content") else str(result)
    
    comm_hub.log_message("code_agent", "mother_agent", "Generated download code")
    return code


# Child Agent: Local File Agent
def clean_filename_for_comparison(filename: str) -> str:
    """Extract the core paper title from filename by removing suffixes and extensions"""
    # Remove file extension
    name_without_ext = filename.replace('.pdf', '')
    
    # Remove common suffixes added by web agent
    suffixes_to_remove = ['_arxiv', '_semantic', '_scholar', '_web', '_researchgate']
    
    for suffix in suffixes_to_remove:
        if name_without_ext.endswith(suffix):
            name_without_ext = name_without_ext[:-len(suffix)]
            break
    
    # Convert underscores back to spaces (reverse of web agent cleaning)
    cleaned = name_without_ext.replace('_', ' ')
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def find_matching_pdf(paper_title: str, references_directory: str, similarity_threshold: float = 0.75) -> str:
    """
    Find PDF file that matches the paper title with improved matching logic
    Returns the filename if found, None otherwise
    """
    if not os.path.exists(references_directory):
        return None
    
    best_match = None
    best_similarity = 0.0
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(references_directory) if f.endswith('.pdf')]
    
    for filename in pdf_files:
        # Extract the clean title from filename
        filename_title = clean_filename_for_comparison(filename)
        
        # Calculate similarity
        similarity = similarity_score(paper_title, filename_title)
        
        # Keep track of the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = filename
    
    # Return the best match if it's above threshold
    if best_similarity >= similarity_threshold:
        return best_match
    
    return None

def local_file_agent_func(paper_title: str) -> str:
    """Process local PDFs or generate summary from title with improved PDF matching"""
    comm_hub = get_comm_hub()
    comm_hub.log_message("local_file_agent", "system", f"Processing: {paper_title}")
    
    # Enhanced PDF search with similarity matching
    found_pdf = find_matching_pdf(paper_title, REFERENCES_DIRECTORY)
    
    if found_pdf:
        comm_hub.log_message("local_file_agent", "system", f"Found matching PDF: {found_pdf}")
        
        # Process the PDF
        try:
            path = os.path.join(REFERENCES_DIRECTORY, found_pdf)
            
            # Verify file exists and is readable
            if not os.path.exists(path):
                comm_hub.log_message("local_file_agent", "system", f"PDF file not found at path: {path}")
                raise FileNotFoundError(f"PDF file not found: {path}")
            
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(path)
            if file_size == 0:
                comm_hub.log_message("local_file_agent", "system", f"PDF file is empty: {found_pdf}")
                raise ValueError(f"PDF file is empty: {found_pdf}")
            
            comm_hub.log_message("local_file_agent", "system", f"Processing PDF: {found_pdf} (size: {file_size} bytes)")
            
            # OCR and process the PDF
            ocr_resp = ocr_pdf(path)
            text = get_combined_text(ocr_resp)
            
            # Check if we got meaningful text
            if not text or len(text.strip()) < 100:
                comm_hub.log_message("local_file_agent", "system", f"Warning: Limited text extracted from PDF: {len(text) if text else 0} characters")
            
            # Generate summary
            summary = generate_summary(text)
            summary_text = summary.content if hasattr(summary, "content") else str(summary)
            
            comm_hub.log_message("local_file_agent", "mother_agent", f"Generated summary from PDF: {found_pdf}")
            return f"[PDF Summary from {found_pdf}] {summary_text}"
            
        except Exception as e:
            comm_hub.log_message("local_file_agent", "system", f"Error processing PDF {found_pdf}: {str(e)}")
            # Fall through to hypothetical summary
    else:
        # Log details about why no PDF was found
        pdf_files = [f for f in os.listdir(REFERENCES_DIRECTORY) if f.endswith('.pdf')] if os.path.exists(REFERENCES_DIRECTORY) else []
        comm_hub.log_message("local_file_agent", "system", f"No matching PDF found for '{paper_title}'")
        comm_hub.log_message("local_file_agent", "system", f"Available PDFs: {pdf_files}")
        
        # Show similarity scores for debugging
        if pdf_files:
            comm_hub.log_message("local_file_agent", "system", "Similarity scores:")
            for pdf_file in pdf_files:
                filename_title = clean_filename_for_comparison(pdf_file)
                score = similarity_score(paper_title, filename_title)
                comm_hub.log_message("local_file_agent", "system", f"  {pdf_file} -> '{filename_title}' (score: {score:.3f})")
    
    # If no PDF found or processing failed, generate hypothetical summary
    comm_hub.log_message("local_file_agent", "system", f"Generating hypothetical summary for: {paper_title}")
    
    prompt = ChatPromptTemplate.from_template(
        """Based on the paper title, provide a brief hypothetical summary of what this research might contain:

Paper Title: {title}

Generate a 3-4 sentence summary describing:
1. The likely research area and objectives
2. Potential methodologies that might be used
3. Expected contributions or findings

Note: This is a hypothetical summary based on the title alone.

Summary:"""
    )
    
    chain = prompt | llm_model
    result = chain.invoke({"title": paper_title})
    summary = result.content if hasattr(result, "content") else str(result)
    
    comm_hub.log_message("local_file_agent", "mother_agent", f"Generated hypothetical summary for: {paper_title}")
    return f"[Hypothetical Summary] {summary}"

# Optional: Add a utility function to list and analyze all PDFs in the directory
def debug_pdf_directory(references_directory: str):
    """Debug function to analyze all PDFs in the directory"""
    if not os.path.exists(references_directory):
        print(f"Directory does not exist: {references_directory}")
        return
    
    pdf_files = [f for f in os.listdir(references_directory) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        filename_title = clean_filename_for_comparison(pdf_file)
        file_path = os.path.join(references_directory, pdf_file)
        file_size = os.path.getsize(file_path)
        print(f"  {pdf_file}")
        print(f"    -> Cleaned title: '{filename_title}'")
        print(f"    -> Size: {file_size} bytes")
        print()

# Mother Agent: Orchestrates child agents
def mother_agent_func(reference_list: List[str]) -> Dict[str, Any]:
    """Mother agent that manages child agents and collects results"""
    comm_hub = get_comm_hub()
    comm_hub.log_message("mother_agent", "system", f"Processing {len(reference_list)} references")
    
    results = {}
    
    for i, ref_title in enumerate(reference_list):
        st.write(f"Processing reference {i+1}/{len(reference_list)}: {ref_title[:50]}...")
        
        # Step 1: Try to download with web agent
        download_result = web_agent_func(ref_title)
        
        if "✅" in download_result:
            # Successfully downloaded, now summarize
            summary = local_file_agent_func(ref_title)
            comm_hub.store_result(ref_title, summary, "Downloaded PDF")
        else:
            # Failed to download, generate hypothetical summary
            code_snippet = code_agent_func(f"Download the paper: {ref_title}")
            comm_hub.log_message("mother_agent", "system",
                         f"Generated code by Code Agent:\n{code_snippet}")

            # 你可以選擇馬上執行這段程式碼，或只是把它存在 logs
            summary = local_file_agent_func(ref_title)
            comm_hub.store_result(ref_title, summary, "Title-based (No PDF)")
        
        # Add small delay to avoid rate limiting
        time.sleep(1)
    
    comm_hub.log_message("mother_agent", "father_agent", f"Completed processing {len(reference_list)} references")
    return comm_hub.get_results_df()


# Father Agent: Critic and orchestrator
def father_agent_func(full_text: str) -> Tuple[List[str], pd.DataFrame]:
    """Father agent that acts as critic and orchestrator"""
    comm_hub = get_comm_hub()
    comm_hub.log_message("father_agent", "system", "Starting reference extraction and processing")
    
    # Extract references from the document
    references = extract_references(full_text)
    
    if not references:
        comm_hub.log_message("father_agent", "system", "No references found in document")
        return [], pd.DataFrame(columns=["Paper Title", "Summary", "Source"])
    
    comm_hub.log_message("father_agent", "mother_agent", f"Found {len(references)} references to process")
    
    # Pass to mother agent for processing
    results_df = mother_agent_func(references)
    
    # Critic function: Evaluate results
    successful = len(results_df[results_df["Source"] == "Downloaded PDF"])
    hypothetical = len(results_df[results_df["Source"] == "Title-based (No PDF)"])
    
    comm_hub.log_message("father_agent", "system", 
                        f"Critique: {successful} PDFs downloaded, {hypothetical} hypothetical summaries generated")
    
    return references, results_df


# --------------------
# 🔄 STREAMLIT STATE
# --------------------

def init_session_state():
    default_state = {
        "indexed": False,
        "chat_history": [],
        "full_text": "",
        "current_pdf": None,
        "summary": "",
        "show_summary": False,
        "question_buffer": "",
        "agent_results": None,
        "extracted_references": [],
    }
    for k, v in default_state.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------
# 🚀 MAIN APP
# ---------------

def main():
    st.set_page_config(layout="wide", page_title="Research Paper RAG with Multi-Agent System")
    init_session_state()

    # ---------- Sidebar ----------
    st.sidebar.header("📄 Upload & Process PDF")
    uploader = st.sidebar.file_uploader("Select PDF file", type=["pdf"])

    if uploader is not None:
        path = os.path.join(PDFS_DIRECTORY, uploader.name)

        if st.session_state.current_pdf != uploader.name:
            st.session_state.indexed = False
            st.session_state.current_pdf = uploader.name
            st.session_state.chat_history = []
            st.session_state.agent_results = None
            st.session_state.extracted_references = []
            if hasattr(get_vector_store(), "store"):
                get_vector_store().store.clear()
            # Reset communication hub
            if "comm_hub" in st.session_state:
                del st.session_state.comm_hub

        if not st.session_state.indexed:
            with open(path, "wb") as f:
                f.write(uploader.read())

            with st.spinner("🔄 Processing OCR and indexing..."):
                try:
                    resp = ocr_pdf(path)
                    captions_lst = extract_images_with_pymupdf(path)
                    
                    pages_txt = resp.pages[:]
                    for cap in captions_lst:
                        pg = cap['page']
                        pages_txt[pg].markdown += (f"\n\n[Figure {cap['figure']}] {cap['caption']}\n")
                    full_text = "\n\n".join(p.markdown for p in pages_txt)
                    st.session_state.full_text = full_text
                    
                    txt_path, md_path = save_ocr_result(full_text, uploader.name, resp)
                    
                    documents = create_documents_with_metadata(full_text, uploader.name)
                    num_chunks = index_documents(documents)
                    st.session_state.indexed = True
                    st.session_state.num_chunks = num_chunks
                    
                    st.sidebar.success("✅ OCR and indexing completed!")
                    st.sidebar.info(f"📄 OCR results saved to:\n- {os.path.basename(txt_path)}\n- {os.path.basename(md_path)}")
                except Exception as e:
                    st.sidebar.error(f"❌ Processing error: {str(e)}")

        with st.sidebar:
            st.subheader("📖 PDF Preview")
            pdf_viewer(input=path, width=300)
            
            st.divider()
            st.subheader("💾 Save Info")
            st.info(f"OCR results auto-save to:\n{SAVE_DIRECTORY}")
            
            # Debug mode for reference extraction
            if st.checkbox("🔧 Debug Reference Extraction", value=False):
                if st.session_state.indexed:
                    with st.expander("Reference Extraction Debug"):
                        # Find references section
                        ref_pattern = r'(?i)(References|Bibliography|Works Cited|Literature Cited)\s*\n+(.*?)(?=\n\n[A-Z]|\n\n\d+\.|\Z)'
                        ref_match = re.search(ref_pattern, st.session_state.full_text, re.DOTALL)
                        
                        if ref_match:
                            st.text_area("Raw References Section (first 2000 chars)", 
                                       ref_match.group(2)[:2000], 
                                       height=200)
                            
                            # Show extracted references
                            refs = extract_references(st.session_state.full_text)
                            st.write(f"**Extracted {len(refs)} references:**")
                            for i, ref in enumerate(refs[:5], 1):  # Show first 5
                                st.write(f"{i}. {ref}")
                            if len(refs) > 5:
                                st.write(f"... and {len(refs) - 5} more")
                        else:
                            st.warning("No references section found in the document")

    # ---------- PAGE STYLING ----------
    st.markdown(
        """
        <style>
    /* Column styling */
    div[data-testid="column"] {
        border-right: 2px solid #e0e0e0;
        padding-right: 20px;
        padding-left: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-testid="stMarkdownContainer"] {
        font-size: 1.1em;
    }
    
    /* Agent message styling with dark mode support */
    .agent-message {
        background-color: rgba(70, 70, 70, 0.8);
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border: 1px solid rgba(150, 150, 150, 0.3);
    }
    
    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .agent-message {
            background-color: rgba(70, 70, 70, 0.8);
            color: #ffffff;
        }
        
        .agent-message strong {
            color: #ffd700;
        }
    }
    
    /* Light mode specific styles */
    @media (prefers-color-scheme: light) {
        .agent-message {
            background-color: #f0f2f6;
            color: #000000;
        }
        
        .agent-message strong {
            color: #0066cc;
        }
    }
    
    /* Fix for Streamlit's dark mode */
    [data-theme="dark"] .agent-message {
        background-color: rgba(70, 70, 70, 0.8) !important;
        color: #ffffff !important;
    }
    
    [data-theme="light"] .agent-message {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
    }
    
    /* Prevent page jumping */
    .main {
        overflow-anchor: auto;
    }
    
    /* Keep focus on current tab */
    .stTabs {
        position: relative;
    }
    
    /* Log container specific styling */
    .log-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        background-color: rgba(40, 40, 40, 0.5);
        border-radius: 5px;
        scroll-behavior: smooth;
    }
    
    /* Timestamp styling */
    .log-timestamp {
        color: #90EE90;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    /* Sender/Receiver styling */
    .log-sender {
        color: #87CEEB;
        font-weight: bold;
    }
    
    .log-receiver {
        color: #FFB6C1;
        font-weight: bold;
    }
    
    /* Message content styling */
    .log-content {
        color: #E0E0E0;
        margin-left: 20px;
    }
    
    /* Success messages */
    .log-success {
        color: #90EE90;
    }
    
    /* Error messages */
    .log-error {
        color: #FF6B6B;
    }
    
    /* Warning messages */
    .log-warning {
        color: #FFD700;
    }
    </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔬 Research Paper RAG with Hierarchical Multi-Agent System")

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary & Chat", "🤖 Agent System", "📚 References", "📊 Agent Logs", "📊 Report"])

    # ----- TAB 1: Summary & Chat -----
    with tab1:
        summary_col, chat_col = st.columns([1, 1])

        with summary_col:
            st.header("📋 Paper Summary")
            if st.session_state.indexed:
                if st.button("🔄 Generate Summary", key="gen_summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(st.session_state.full_text)
                        st.session_state.summary = summary
                        st.session_state.show_summary = True
                        st.rerun()
                        
                if st.session_state.show_summary:
                    st.write(st.session_state.summary)
                    
                chunk_total = len(get_vector_store().store) if hasattr(get_vector_store(), "store") else "N/A"
                st.info(
                    f"""
                    📊 Document Statistics:
                    - File name: {st.session_state.current_pdf}
                    - Total characters: {len(st.session_state.full_text)}
                    - Number of chunks: {chunk_total}
                    """
                )
            else:
                st.info("📤 Please upload and index a PDF via the sidebar to see a summary.")

        with chat_col:
            st.header("💬 Chat with Document")

            if st.session_state.indexed:
                st.subheader("💡 Suggested Questions")
                suggestions = [
                    "What is the main research objective?",
                    "What methodology was used?",
                    "What are the key findings and conclusions?",
                    "What are the limitations of this study?",
                    "What are the future research directions?",
                ]
                
                def set_question(q):
                    st.session_state.question_buffer = q
                    
                for i, sug in enumerate(suggestions):
                    st.button(sug, key=f"sug_top_{i}", on_click=set_question, args=(sug,))
                st.divider()

            question = st.text_input("Enter your question:", value=st.session_state.question_buffer, key="question_input")
            st.session_state.question_buffer = question

            for idx, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Answer:** {chat['answer']}")
                    if chat.get("sources"):
                        with st.expander("View Sources"):
                            for i, src in enumerate(chat["sources"][:3]):
                                st.write(f"Source {i+1}: {src.page_content[:200]} …")
                    st.divider()

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Ask Question", type="primary", use_container_width=True):
                    if not st.session_state.indexed:
                        st.warning("⚠️ Please upload and index a PDF first via the sidebar.")
                    elif not question:
                        st.warning("⚠️ Please enter a question.")
                    else:
                        with st.spinner("🤔 Thinking..."):
                            result = answer_question_with_rag(question)
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                            })
                            st.session_state.question_buffer = ""
                            st.rerun()

            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.question_buffer = ""
                    st.rerun()

    # ----- TAB 2: Agent System -----
    with tab2:
        st.header("🤖 Hierarchical Multi-Agent System")
        
        if st.session_state.indexed:
            st.markdown("""
            ### Agent Hierarchy:
            - **👨‍💼 Father Agent**: Critic & Orchestrator
            - **👩‍💼 Mother Agent**: Creates child agents and manages communication
            - **👶 Child Agents**:
                - 🌐 Web Agent: Downloads from web, arXiv, ResearchGate
                - 💻 Code Agent: Generates download code
                - 📄 Local File Agent: Processes PDFs and generates summaries
            """)
            
            if st.button("🚀 Extract and Process All References", type="primary"):
                with st.spinner("Father Agent orchestrating reference extraction..."):
                    # Clear previous results
                    if "comm_hub" in st.session_state:
                        del st.session_state.comm_hub
                    
                    # Run father agent
                    references, results_df = father_agent_func(st.session_state.full_text)
                    
                    st.session_state.extracted_references = references
                    st.session_state.agent_results = results_df
                    
                    st.success(f"✅ Processed {len(references)} references!")
                    st.rerun()
            
            if st.session_state.extracted_references:
                with st.expander("📚 View Extracted References", expanded=True):
                    st.write(f"**Total references found:** {len(st.session_state.extracted_references)}")
                    
                    # Display in a nice format
                    for i, ref in enumerate(st.session_state.extracted_references, 1):
                        col1, col2 = st.columns([1, 10])
                        with col1:
                            st.write(f"**{i}.**")
                        with col2:
                            st.write(ref)
                    
                    # Option to download the reference list
                    ref_text = "\n".join([f"{i}. {ref}" for i, ref in enumerate(st.session_state.extracted_references, 1)])
                    st.download_button(
                        label="📥 Download Reference List",
                        data=ref_text,
                        file_name=f"references_{st.session_state.current_pdf.replace('.pdf', '')}.txt",
                        mime="text/plain"
                    )
            
            if st.session_state.agent_results is not None and not st.session_state.agent_results.empty:
                st.subheader("📊 Reference Processing Results")
                st.dataframe(st.session_state.agent_results, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_refs = len(st.session_state.agent_results)
                    st.metric("Total References", total_refs)
                with col2:
                    downloaded = len(st.session_state.agent_results[st.session_state.agent_results["Source"] == "Downloaded PDF"])
                    st.metric("PDFs Downloaded", downloaded)
                with col3:
                    hypothetical = len(st.session_state.agent_results[st.session_state.agent_results["Source"] == "Title-based (No PDF)"])
                    st.metric("Hypothetical Summaries", hypothetical)
                
                # Export option
                csv = st.session_state.agent_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"reference_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📤 Please upload and index a PDF to use the agent system.")

    # ----- TAB 3: References -----
    with tab3:
        st.header("📚 Downloaded References")
        
        ref_files = os.listdir(REFERENCES_DIRECTORY)
        if ref_files:
            st.write(f"Found {len(ref_files)} reference files:")
            
            for i, filename in enumerate(ref_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i+1}. {filename}")
                with col2:
                    if st.button("📖 View", key=f"view_ref_{i}"):
                        file_path = os.path.join(REFERENCES_DIRECTORY, filename)
                        st.info(f"Viewing: {filename}")
                        pdf_viewer(input=file_path, width=700)
        else:
            st.info("No reference files found. Use the Agent System to download references.")

    # ----- TAB 4: Agent Logs -----
    with tab4:
        st.header("📊 Agent Communication Logs")
        
        if "comm_hub" in st.session_state:
            comm_hub = get_comm_hub()
            
            if comm_hub.messages:
                st.subheader("Message Log")
                for msg in reversed(comm_hub.messages[:]):  # Show last 20 messages
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                    st.markdown(
                        f'<div class="agent-message">'
                        f'<strong>[{timestamp}] {msg["sender"]} → {msg["receiver"]}:</strong> '
                        f'{msg["message"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            if comm_hub.failed_downloads:
                st.subheader("❌ Failed Downloads")
                for title in comm_hub.failed_downloads:
                    st.write(f"- {title}")
        else:
            st.info("No agent logs available. Run the agent system to see communication logs.")

    # ----- TAB 5: Report -----
    with tab5:
        st.header("📊 Comprehensive Research Report")
        
        if st.session_state.indexed:
            # Check if we have all necessary components
            has_summary = st.session_state.get('summary', '') != ''
            has_references = st.session_state.get('agent_results', None) is not None
            
            # Status indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.indexed:
                    st.success("✅ PDF Processed")
                else:
                    st.warning("❌ PDF Not Processed")
            with col2:
                if has_summary:
                    st.success("✅ Summary Generated")
                else:
                    st.warning("❌ Summary Not Generated")
            with col3:
                if has_references:
                    st.success("✅ References Analyzed")
                else:
                    st.warning("❌ References Not Analyzed")
            
            st.divider()
            
            # Instructions if components are missing
            if not has_summary:
                st.info("📝 Please generate a summary in Tab 1 first.")
            if not has_references:
                st.info("🤖 Please run the Agent System in Tab 2 to analyze references.")
            
            # Generate report button
            if st.button("📄 Generate Comprehensive Report", type="primary", disabled=not (has_summary and has_references)):
                with st.spinner("Generating comprehensive report... This may take a moment."):
                    try:
                        # Generate the report
                        report = generate_comprehensive_report(
                            full_text=st.session_state.full_text,
                            summary=st.session_state.summary.content if hasattr(st.session_state.summary, 'content') else str(st.session_state.summary),
                            agent_results_df=st.session_state.agent_results,
                            pdf_name=st.session_state.current_pdf
                        )
                        
                        # Save the report
                        report_path = save_report(report, st.session_state.current_pdf)
                        
                        # Store in session state
                        st.session_state.comprehensive_report = report
                        st.session_state.report_path = report_path
                        
                        st.success(f"✅ Report generated and saved to: {os.path.basename(report_path)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.exception(e)
            
            # Display report if available
            if 'comprehensive_report' in st.session_state:
                st.divider()
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=st.session_state.comprehensive_report,
                        file_name=f"report_{st.session_state.current_pdf.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Convert to HTML for better viewing
                    html_report = markdown.markdown(st.session_state.comprehensive_report, extensions=['tables', 'fenced_code'])
                    st.download_button(
                        label="📥 Download Report (HTML)",
                        data=html_report,
                        file_name=f"report_{st.session_state.current_pdf.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
                
                st.divider()
                
                # Display the report
                st.markdown(st.session_state.comprehensive_report)
                
        else:
            st.info("📤 Please upload and process a PDF first to generate a comprehensive report.")
            st.markdown("""
            ### How to Generate a Complete Report:
            
            1. **Upload a PDF** in the sidebar
            2. **Generate Summary** in Tab 1
            3. **Run Agent System** in Tab 2 to analyze references
            4. **Come back here** to generate the comprehensive report
            
            The report will include:
            - Executive summary
            - Detailed paper structure analysis
            - Reference analysis with summaries
            - Key insights and connections
            - Recommendations for further research
            - Document statistics
            """)


if __name__ == "__main__":
    main()