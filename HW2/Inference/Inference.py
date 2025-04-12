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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_predictions(predictions, ground_truth, model_name="Model"):
    """
    Evaluate predictions against ground truth
    predictions: List of predicted labels (0 or 1)
    ground_truth: List of true labels (0 or 1)
    """
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions, zero_division=0)
    rec = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Print results
    print(f"\n[{model_name} Evaluation Results]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print("               Predicted")
    print("              | Neg | Pos |")
    if cm.shape == (2, 2):
        print(f"Actual Neg | {cm[0][0]:3d} | {cm[0][1]:3d} |")
        print(f"Actual Pos | {cm[1][0]:3d} | {cm[1][1]:3d} |")
    else:
        print("Invalid confusion matrix shape")
    
    # Plot and save confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Ensure the pred directory exists
    os.makedirs('pred', exist_ok=True)
    
    # Save the plot with model name in the filename
    plt.tight_layout()
    cm_filename = f"pred/confusion_matrix_{model_name.lower()}.png"
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")
    plt.close()
    
    # Return metrics in a dictionary
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }
def preprocess_test_data(test_csv_path):
    """
    Load and preprocess the test data
    """
    # Load the data
    print(f"Loading data from {test_csv_path}")
    test_data = pd.read_csv(test_csv_path)
    
    # Print available columns to help with debugging
    print(f"Available columns: {test_data.columns.tolist()}")
    
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
    
    return test_data['clean_text'], test_data[target_column], test_data

def main():
    parser = argparse.ArgumentParser(description='Inference script for tweet sentiment model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--test', type=str, required=True, help='Path to the test dataset CSV')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru'], 
                        help='Type of model to use: lstm or gru')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--eval', type=str, help='Path to evaluation dataset CSV (optional)')
    
    args = parser.parse_args()
    
    # Check if the test file exists
    if not os.path.isfile(args.test):
        print(f"Error: Test file {args.test} does not exist")
        return
    
    # If model is a directory, find the model file
    model_path = args.model
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        if not model_files:
            print(f"Error: No model files found in {model_path}")
            return
        
        # Try to find a model file matching the specified type
        model_type_files = [f for f in model_files if args.model_type.upper() in f]
        if model_type_files:
            model_path = os.path.join(model_path, model_type_files[0])
        else:
            model_path = os.path.join(model_path, model_files[0])
            
        print(f"Using model file: {model_path}")
    
    # Load and preprocess the test data
    print("Loading and preprocessing test data...")
    X_test, y_test, test_df = preprocess_test_data(args.test)
    X_test = X_test.tolist()
    y_test = y_test.tolist()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create test dataset and loader
    test_dataset = TweetBertDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load the model
    print(f"Loading {args.model_type.upper()} model from {model_path}...")
    model = load_model(model_path, args.model_type)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Create output directory if it doesn't exist
    output_dir = 'pred'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on model type
    output_filename = f"result_{args.model_type.lower()}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save predictions
    test_df['target'] = predictions
    if 'clean_text' in test_df.columns:
        test_df = test_df.drop('clean_text', axis=1)
    if 'text' in test_df.columns:
        test_df = test_df.drop('text', axis=1)
    if 'location' in test_df.columns:
        test_df = test_df.drop('location', axis=1)
    if 'keyword' in test_df.columns:
        test_df = test_df.drop('keyword', axis=1)
    print(f"Saving predictions to {output_path}")
    test_df.to_csv(output_path, index=False)
    
    # Evaluate the model if evaluation data is provided
    if args.eval:
        if not os.path.isfile(args.eval):
            print(f"Warning: Evaluation file {args.eval} does not exist. Skipping evaluation.")
        else:
            print(f"\nEvaluating predictions against {args.eval}...")
            
            # Load the ground truth data
            try:
                eval_df = pd.read_csv(args.eval)
                print(f"Evaluation data columns: {eval_df.columns.tolist()}")
                
                # Find target column in evaluation data
                eval_target_column = None
                for col in ['target', 'label', 'sentiment', 'class']:
                    if col in eval_df.columns:
                        eval_target_column = col
                        print(f"Using '{col}' as ground truth column")
                        break
                
                if eval_target_column is None:
                    print("Error: No target column found in evaluation data")
                else:
                    # Ensure the IDs match between test and eval data if IDs are present
                    if 'id' in test_df.columns and 'id' in eval_df.columns:
                        test_df = test_df.sort_values('id')
                        eval_df = eval_df.sort_values('id')
                        
                        if not test_df['id'].equals(eval_df['id']):
                            print("Warning: ID columns in test and evaluation data do not match")
                    
                    # Get the predicted and ground truth labels
                    y_pred = test_df['target'].tolist()
                    y_true = eval_df[eval_target_column].tolist()
                    
                    # Evaluate
                    metrics = evaluate_predictions(y_pred, y_true, model_name=args.model_type.upper())
                    
                    # Save evaluation metrics to a file
                    metrics_filename = f"metrics_{args.model_type.lower()}.txt"
                    metrics_path = os.path.join(output_dir, metrics_filename)
                    
                    with open(metrics_path, 'w') as f:
                        f.write(f"Model: {args.model_type.upper()}\n")
                        f.write(f"Evaluation Dataset: {args.eval}\n")
                        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                        f.write(f"Precision: {metrics['precision']:.4f}\n")
                        f.write(f"Recall: {metrics['recall']:.4f}\n")
                        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                        f.write("\nConfusion Matrix:\n")
                        f.write("              | Predicted Neg | Predicted Pos |\n")
                        f.write(f"Actual Neg | {metrics['confusion_matrix'][0][0]:13d} | {metrics['confusion_matrix'][0][1]:13d} |\n")
                        f.write(f"Actual Pos | {metrics['confusion_matrix'][1][0]:13d} | {metrics['confusion_matrix'][1][1]:13d} |")
                        
                    print(f"Evaluation metrics saved to {metrics_path}")
            
            except Exception as e:
                print(f"Error during evaluation: {e}")
    
    print("\nInference complete!")

if __name__ == "__main__":
    main()