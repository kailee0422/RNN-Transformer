# Requirements

Before running this program, please install the required packages:

```bash
pip install -r requirements.txt
```

# How to Run Training Program

You can choose to run this program **locally in VS Code** or **directly on Google Colab**.


## 🖥️ Option 1: Run in VS Code (Locally)

1. **Set Up Environment**  
   Make sure you have Python installed and a virtual environment activated.

2. **Install Required Packages**  
   Use the following command to install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**  
   Start Jupyter with:

   ```bash
   jupyter notebook
   ```

4. **Run the Notebook**  
   - Open `main.ipynb`
   - Run each cell sequentially

5. **View Results**  
   The notebook will classify texts as either AI-generated or human-written, and display accuracy metrics along with visualizations.



## ☁️ Option 2: Run on Google Colab

Click the badge below to open and run the notebook on Colab (no local setup required):

<a target="_blank" href="https://colab.research.google.com/github/kailee0422/RNN-Transformer/blob/main/HW2/main.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



# How to Run Inference 

If you want to make predictions using a trained model, you can use the `inference.py` or `inference_assignment.py` script running in cmd or VS Code.The main difference between these two files is the output format. Check below for more details.

##  inference.py

###  Purpose
This script loads a saved model and runs inference on a test dataset. Optionally, it can also evaluate results against ground-truth labels.

### ✅ Example Usage

```bash
python inference.py --model ./model --test "./Test data/test.csv" --model_type lstm --eval "./Test data/test_ans.csv"
```

### 🔧 Arguments Explained

| Argument | Description |
|----------|-------------|
| `--model` | **(Required)** Path to the directory containing the saved model. |
| `--test` | **(Required)** Path to the test dataset CSV file. |
| `--model_type` | *(Optional)* Choose the model architecture to use: `'lstm'` or `'gru'`. Default is `'lstm'`. |
| `--batch_size` | *(Optional)* Number of samples processed per batch during inference. Default is `32`. |
| `--eval` | *(Optional)* Path to evaluation CSV file (with ground-truth labels). If provided, the script will compute evaluation metrics (e.g., accuracy). |

### 📄 Output Format

- The output CSV will contain two columns:
  - `id`: The unique identifier of the input sample.
  - `predicted`: The predicted label (`0` for human-written, `1` for AI-generated).


##  inference_assignment.py

###  Purpose
This script loads a saved model and runs inference on a test dataset.

### ✅ Example Usage

```bash
python inference_assignment.py --model ./model --test "./Test data/test.csv" "
```

### 🔧 Arguments Explained

| Argument | Description |
|----------|-------------|
| `--model` | **(Required)** Path to the directory containing the saved model. |
| `--test` | **(Required)** Path to the test dataset CSV file. |
| `--batch_size` | *(Optional)* Number of samples processed per batch during inference. Default is `32`. |


### 📄 Output Format

- The output CSV will contain two columns:
  - `id`: The unique identifier of the input sample.
  - `LSTM`: The predicted label by LSTM(`0` for human-written, `1` for AI-generated).
  - `GRU`: The predicted label by GRU(`0` for human-written, `1` for AI-generated).


## 💬 Need Help?

If you encounter any issues, feel free to contact [me](mailto:aa34239387@gmail.com) for assistance.

---

需要我幫你也整理一下 `inference_assignment.py` 的用法嗎？或者你有其他的程式要一起寫進說明文件裡？
