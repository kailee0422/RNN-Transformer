## 🔍 Overview

This project explores the use of **Compared with LSTM and GRU** for classifying whether a tweet is about a real disaster or not. It is based on the [Kaggle NLP Disaster Tweets dataset](https://www.kaggle.com/competitions/nlp-getting-started/overview). The goal is to compare the performance of LSTM and GRU models in terms of classification accuracy, training speed, and resource usage.

## Table of Contents
- [Model Pipeline](#model-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)

## Model Pipeline

### 1. **Data Preprocessing**
- Remove punctuation, HTML, and non-ASCII characters
- Normalize URLs, usernames, numbers, emojis, and common abbreviations
- Token replacements for consistent input to the models

### 2. **Model Architecture**
- **BERT-base-uncased** encoder (frozen during training)
- One-layer **LSTM** or **GRU** (hidden size: 256)
- Dropout layer (0.2)
- Fully connected layer + Sigmoid activation for binary output

### 3. **Training Setup**
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.0001)
- Scheduler: ReduceLROnPlateau
- Epochs: 30, Batch size: 32
- Validation split: 20%


## Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1 Score**
- Training time and GPU memory usage also recorded
- Confusion matrices and training/validation curves provided for analysis



## Results

### Loss and Accuracy Plots
<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Figure1.png" width="60%"/>

<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Figure2.png" width="60%"/>

###  Validation Set

<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Table1.png" width="60%"/>


###  Test Set

<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Table2.png" width="60%"/>
<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Figure3.png" width="60%"/>

⚠️ *Due to the test set containing no positive samples, both models failed to predict any positive cases, leading to precision/recall = 0.*

###  Resource Usage

<img src="https://raw.githubusercontent.com/kailee0422/RNN-Transformer/main/HW2/Picture/Table3.png" width="60%"/>


## Conclusion

GRU achieved higher accuracy and precision compared to LSTM, while also training slightly faster with fewer parameters. Although both models failed to identify positive samples in the test set, GRU showed better performance and efficiency overall, making it more suitable for this binary classification task.






