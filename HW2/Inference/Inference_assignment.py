import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import argparse
import os
import re
import string
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load abbreviations from file
def load_abbreviations(file_path):
    """
    Load abbreviations from a text file
    Format: abbreviation expansion
    Example: "$ dollar"
    """
    abbr_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split(' ', 1)  # Split at the first space
                    if len(parts) == 2:
                        abbr, expansion = parts
                        abbr_dict[abbr.strip()] = expansion.strip()
        print(f"Loaded {len(abbr_dict)} abbreviations from {file_path}")
    except Exception as e:
        print(f"Error loading abbreviations file: {e}")
        print("Using default abbreviations dictionary")
        return {
            "$" : " dollar ",
            "€" : " euro ",
            "lol" : "laughing out loud",
            "wtf" : "what the fuck",
            "omg" : "oh my god"
        }
    return abbr_dict

# Try to load abbreviations from file, otherwise use default
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abbr_file = os.path.join(script_dir, 'abbreviations.txt')
    abbreviations = load_abbreviations(abbr_file)
except:
    abbreviations = {
        "$" : " dollar ",
        "€" : " euro ",
        "lol" : "laughing out loud",
        "wtf" : "what the fuck",
        "omg" : "oh my god"
    }
    print("Using default abbreviations dictionary")

# Text preprocessing functions
def toclean_text(text):
    clean_text = [char for char in text if char not in string.punctuation]
    clean_text = ''.join(clean_text)
    return clean_text

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'URL', text)

def remove_HTML(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_not_ASCII(text):
    text = ''.join([word for word in text if word in string.printable])
    return text

def word_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

def replace_abbrev(text):
    string = ""
    for word in text.split():
        string += word_abbrev(word) + " "        
    return string

def remove_mention(text):
    at = re.compile(r'@\S+')
    return at.sub(r'USER', text)

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text)

def transcription_sad(text):
    smiley = re.compile(r'[8:=;][\'\-]?[(\\/]')
    return smiley.sub(r'SADFACE', text)

def transcription_smile(text):
    smiley = re.compile(r'[8:=;][\'\-]?[)dDp]')
    return smiley.sub(r'SMILE', text)

def transcription_heart(text):
    heart = re.compile(r'<3')
    return heart.sub(r'HEART', text)

def clean_tweet(text):
    # Remove non text
    text = remove_URL(text)
    text = remove_HTML(text)
    text = remove_not_ASCII(text)
    
    # replace abbreviations, @ and number
    text = replace_abbrev(text)  
    text = remove_mention(text)
    text = remove_number(text)
    
    # Remove emojis / smileys
    text = remove_emoji(text)
    text = transcription_sad(text)
    text = transcription_smile(text)
    text = transcription_heart(text)
  
    return text

# Dataset class
class TweetBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        """
        texts: List[str], each entry is a sentence
        labels: List[int] or numpy array, each entry is 0/1
        tokenizer: BertTokenizer
        max_len: the length to pad/truncate to
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'  # return PyTorch tensor
        )
        # encoding contains: input_ids, token_type_ids, attention_mask
        # shape is (1, max_len), so use squeeze(0) to make it (max_len,)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        return input_ids, attention_mask, token_type_ids, label

# Model definitions
class BertLSTM(nn.Module):
    def __init__(self, lstm_hidden=256, dropout=0.4, freeze_bert=True):
        """
        lstm_hidden: Hidden vector dimension of LSTM output
        dropout: Dropout parameter for LSTM
        freeze_bert: Whether to freeze BERT parameters (True=do not train BERT)
        """
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768,         # BERT-base hidden size
            hidden_size=lstm_hidden,
            batch_first=True,
            dropout=dropout,
            num_layers=1
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(lstm_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state

        lstm_out, (h_n, c_n) = self.lstm(last_hidden_state)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(1)  # Convert to (batch_size,)

class BertGRU(nn.Module):
    def __init__(self, gru_hidden=256, dropout=0.4, freeze_bert=True):
        super(BertGRU, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(
            input_size=768,  # BERT-base hidden size
            hidden_size=gru_hidden,
            batch_first=True,
            dropout=dropout,
            num_layers=1
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(gru_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state

        gru_out, h_n = self.gru(last_hidden_state)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(1)

def load_model(model_path, model_type="lstm"):
    """
    Load a pre-trained model
    model_path: Path to the saved model state dict
    model_type: either "lstm" or "gru"
    """
    if model_type.lower() == "lstm":
        model = BertLSTM(lstm_hidden=256, dropout=0.4, freeze_bert=True)
    elif model_type.lower() == "gru":
        model = BertGRU(gru_hidden=256, dropout=0.4, freeze_bert=True)
    else:
        raise ValueError("Model type must be either 'lstm' or 'gru'")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def generate_predictions(model, test_loader):
    """
    Generate predictions from a model without evaluating metrics
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, _ in tqdm(test_loader, desc="Generating predictions"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            preds = (outputs >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
    
    return all_preds

def preprocess_test_data(test_csv_path):
    """
    Load and preprocess the test data
    """
    # Load the data
    print(f"Loading test data from {test_csv_path}")
    test_data = pd.read_csv(test_csv_path)
    
    # Print available columns to help with debugging
    print(f"Available columns in the test data: {test_data.columns.tolist()}")
    
    # Figure out which column is the target column (label)
    target_column = None
    possible_target_columns = ['target', 'label', 'sentiment', 'class', 'y']
    
    for col in possible_target_columns:
        if col in test_data.columns:
            target_column = col
            print(f"Using '{col}' as the target column")
            break
    
    if target_column is None:
        print("Warning: No recognized target column found. Assuming all samples have label 0.")
        test_data['target'] = 0  # Create a dummy target column
        target_column = 'target'
    
    # Get the text column (assuming it's called 'text')
    text_column = 'text'
    if text_column not in test_data.columns:
        possible_text_columns = ['text', 'tweet', 'content', 'message', 'review']
        for col in possible_text_columns:
            if col in test_data.columns:
                text_column = col
                print(f"Using '{col}' as the text column")
                break
        
        if text_column not in test_data.columns:
            raise ValueError(f"No text column found in {test_csv_path}. Expected one of: {possible_text_columns}")
    
    # Apply text cleaning
    print("Preprocessing text...")
    test_data['clean_text'] = test_data[text_column].apply(toclean_text)
    test_data["clean_text"] = test_data["clean_text"].apply(clean_tweet)
    
    # Check for ID column
    id_column = None
    possible_id_columns = ['id', 'ID', 'tweet_id', 'tweetid', 'index']
    
    for col in possible_id_columns:
        if col in test_data.columns:
            id_column = col
            print(f"Using '{col}' as the ID column")
            break
    
    if id_column is None:
        print("No ID column found. Creating index column as ID.")
        test_data['id'] = test_data.index
        id_column = 'id'
    
    return test_data['clean_text'], test_data[target_column], id_column, test_data

def find_model_file(model_dir, model_type):
    """
    Find model file in the directory that matches the model type
    """
    if not os.path.isdir(model_dir):
        return model_dir  # Assume it's a direct file path
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")
    
    # Try to find a model file matching the specified type
    model_type_files = [f for f in model_files if model_type.upper() in f]
    if model_type_files:
        return os.path.join(model_dir, model_type_files[0])
    else:
        # If no model with specific type is found, use any model file
        return os.path.join(model_dir, model_files[0])

def main():
    parser = argparse.ArgumentParser(description='Inference script for tweet sentiment models (LSTM & GRU)')
    parser.add_argument('--model', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--test', type=str, required=True, help='Path to the test dataset CSV')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Check if the test file exists
    if not os.path.isfile(args.test):
        print(f"Error: Test file {args.test} does not exist")
        return
    
    # Load and preprocess the test data
    print("Loading and preprocessing test data...")
    X_test, y_test, id_column, test_df = preprocess_test_data(args.test)
    X_test = X_test.tolist()
    y_test = y_test.tolist()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create test dataset and loader
    test_dataset = TweetBertDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Find and load the LSTM model
    try:
        lstm_model_path = find_model_file(args.model, "lstm")
        print(f"Loading LSTM model from {lstm_model_path}...")
        lstm_model = load_model(lstm_model_path, "lstm")
        
        # Generate LSTM predictions
        print("Generating LSTM predictions...")
        lstm_predictions = generate_predictions(lstm_model, test_loader)
    except Exception as e:
        print(f"Error loading or running LSTM model: {e}")
        print("Using zeros for LSTM predictions")
        lstm_predictions = [0] * len(y_test)
    
    # Find and load the GRU model
    try:
        gru_model_path = find_model_file(args.model, "gru")
        print(f"Loading GRU model from {gru_model_path}...")
        gru_model = load_model(gru_model_path, "gru")
        
        # Generate GRU predictions
        print("Generating GRU predictions...")
        gru_predictions = generate_predictions(gru_model, test_loader)
    except Exception as e:
        print(f"Error loading or running GRU model: {e}")
        print("Using zeros for GRU predictions")
        gru_predictions = [0] * len(y_test)
    
    # Create output directory if it doesn't exist
    output_dir = 'pred'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    output_filename = "result_assignment.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create output DataFrame with ID, LSTM predictions, and GRU predictions
    result_df = pd.DataFrame({
        'ID': test_df[id_column],
        'LSTM': lstm_predictions,
        'GRU': gru_predictions
    })
    
    print(f"Saving predictions to {output_path}")
    result_df.to_csv(output_path, index=False)
    
    print("\nInference complete!")

if __name__ == "__main__":
    main()