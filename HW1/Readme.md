# HW1 Experiment Report

## Table of Contents
- [Introduction](#introduction)
- [Method](#method)
  - [Word Embedding Techniques](#word-embedding-techniques)
  - [Model](#model)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Introduction
This experiment aims to classify AI-generated versus human-written texts using LSTM-based models. The task includes data preprocessing, embedding techniques exploration, model training, and evaluation.

---

## Method
### Word Embedding Techniques
- **BERT (cased):** Transformer-based contextual embeddings (768-dimensions).
- **AutoTokenizer & AutoEmbedding (BERT uncased):** Automatic tokenizer and embedding selection; converts all text to lowercase.
- **Basic Custom Embedding:** Vocabulary built manually; embeddings initialized randomly and learned during training (512 tokens max).
- **Word2Vec:** Captures local context (300-dimensions).
- **FastText:** Extends Word2Vec by character n-grams (300-dimensions).

### Model Architectures
- **LSTM:** Captures sequential dependencies.
- **CNN-LSTM:** Combines CNN local feature extraction with LSTM sequential modeling.
- **Bidirectional LSTM (BiLSTM):** Processes text in both forward and backward directions.

---

## Results
![image](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Table1.png)

- **Figure 1:**
![image](https://github.2com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure1.png)
- Training and testing loss (left), test accuracy (right) for Basic embedding with LSTM.
- **Figure 2:**
![image](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure2.png)
- Training and testing loss (left), test accuracy (right) for Basic embedding with CNN-LSTM.
- **Figure 3:**
![image](https://github.com/kailee0422/RNN-Transformer/blob/main/HW1/picture/Figure3.png)
- Training and testing loss (left), test accuracy (right) for Basic embedding with BiLSTM.

---

## Conclusion
The experiment revealed that all embedding techniques achieved over 98% accuracy, with Basic embedding unexpectedly outperforming advanced pre-trained methods, indicating simpler embeddings may sometimes better align with specific datasets or model structures. Further exploration through cross-validation and additional datasets is suggested.

