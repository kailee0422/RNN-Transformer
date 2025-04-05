# HW1 Experiment Report

## Table of Contents
- [Introduction](#introduction)
- [Method](#method)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction
This experiment aims to classify AI-generated versus human-written texts using LSTM-based models. The task includes data preprocessing, embedding techniques exploration, model training, and evaluation.


## Method
### Word Embedding Techniques
- **BERT (cased):** Transformer-based contextual embeddings (768-dimensions).

  ```bash
  from transformers import BertTokenizer, BertModel
  ```
- **AutoTokenizer & AutoEmbedding (BERT uncased):** Automatic tokenizer and embedding selection; converts all text to lowercase.

  ```bash
  from transformers import AutoTokenizer, AutoModel
  ```
- **Basic Custom Embedding:** Vocabulary built manually; embeddings initialized randomly and learned during training (512 tokens max).
- **Word2Vec:** Captures local context (300-dimensions).

  [Download pre-trained model here](https://huggingface.co/fse/word2vec-google-news-300)
- **FastText:** Extends Word2Vec by character n-grams (300-dimensions).

  [Download pre-trained model here](https://fasttext.cc/docs/en/english-vectors.html)

### Model Architectures
- **LSTM:** Captures sequential dependencies.
- **CNN-LSTM:** Combines CNN local feature extraction with LSTM sequential modeling.
- **Bidirectional LSTM (BiLSTM):** Processes text in both forward and backward directions.



## Results

- **Table 1:** Test Accuracy for Different Tokenizer and Word Embedding Combinations.
<br><br> 
![Test Accuracy Table](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Table1.png)
<br><br>
- **Figure 1:** Training and testing loss (left), test accuracy (right) for Basic embedding with LSTM.
<br><br>  
![Figure 1](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure1.png)
<br><br>
- **Figure 2:** Training and testing loss (left), test accuracy (right) for Basic embedding with CNN-LSTM.
<br><br> 
![Figure 2](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure2.png)
<br><br>
- **Figure 3:** Training and testing loss (left), test accuracy (right) for Basic embedding with BiLSTM.
<br><br> 
![Figure 3](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure3.png)



## Conclusion
The experiment revealed that all embedding techniques achieved over 98% accuracy, with Basic embedding unexpectedly outperforming advanced pre-trained methods, indicating simpler embeddings may sometimes better align with specific datasets or model structures. Further exploration through cross-validation and additional datasets is suggested.

