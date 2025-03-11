# 🎬 IMDB Movie Review Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews using a **Simple Recurrent Neural Network (RNN)**. It includes data preprocessing, model training, and a **Streamlit-based web application** for real-time predictions.

---

### 📁 Project Structure

📂 IMDB-Sentiment-Analysis

📜 embedding.ipynb       # Word embeddings analysis

📜 simplernn.ipynb       # Building & training the Simple RNN model

📜 prediction.ipynb      # Predicting sentiment using the trained model

📜 main.py               # Streamlit app for real-time sentiment classification

📜 simple_rnn_imdb.h5    # Pre-trained RNN model

📜 requirements.txt      # Dependencies required to run the project

📜 README.md             # Project documentation

---

### 📝 Description

This project classifies IMDB movie reviews as **positive** or **negative** using a **Simple RNN model trained on the IMDB dataset**. 

It includes:

✅ **Data Preprocessing:** Tokenization, padding, and encoding of text data.

✅ **RNN Model Training:** A neural network trained using TensorFlow/Keras.

✅ **Prediction & Interpretation:** Classifies reviews using the trained model.

✅ **Web Interface:** A **Streamlit** application for real-time sentiment classification.

---

## 📊 Dataset

- **Source:** [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 movie reviews (25K training, 25K testing)
- **Labels:** Binary classification (Positive / Negative)

### 🔹 Data Preprocessing

1. Convert text to lowercase and tokenize words.
2. Encode words using the IMDB dataset word index.
3. Apply **sequence padding** to ensure uniform input size (max length = **500**).

---

## 🏗 Model Architecture

- **Embedding Layer**: Converts words into dense vector representations.
- **Simple RNN Layer**: Captures sequential dependencies in the text.
- **Dense Layer (ReLU Activation)**: Processes extracted features.
- **Output Layer (Sigmoid Activation)**: Predicts sentiment score.

### 🔹 Training Details

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 64
- **Epochs:** 10
- **Validation Split:** 20%

---

### 🚀 How to Run & Some other Help

#### 1️⃣ To Create a venv environment

Command = **conda create -p venv python==3.11 -y**  [You can use any version of python as you need]

#### 2️⃣ For Activating and Deactivating the venv environment

Activate - **conda activate**
Deactivate - **conda deactivate**

#### 3️⃣ To Install the requirements.txt

Command - **pip install -r requirements.txt**

#### 4️⃣ Run the Streamlit App

Command - **streamlit run main.py**

---

### 🎭 Example Usage

**Input Review:**

\"This movie was absolutely fantastic! The acting was top-notch and the storyline was gripping.\"

**Predicted Sentiment:**
✅ **Positive** (Prediction Score: **0.92**)

---

### 📌 Features

✔️ **Pre-trained RNN Model** for sentiment classification.

✔️ **Fast and Accurate** predictions on movie reviews.

✔️ **User-friendly Web App** built using Streamlit.

✔️ **Real-time Inference** for user input reviews.

---

## 📌 Author

👨‍💻 **Vibhu Pratap**

🔗 **GitHub:** [yourgithub](https://github.com/vibhupratap-007)

🔗 **LinkedIn:** [yourlinkedin](https://www.linkedin.com/in/vibhu-pratap-v/)

---
