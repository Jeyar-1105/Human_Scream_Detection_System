
# 🎤 Human Scream Detection using Deep Learning

This project detects and classifies human scream audio into three distinct categories using deep learning. It uses **MFCC audio features**, and models like **CNN** and **LSTM**, trained on a custom dataset. The project includes a **Flask web interface** for real-time audio upload and prediction.

---

## 🔍 Project Overview

Human scream detection plays a key role in safety, surveillance, and emotion recognition. This system classifies uploaded `.wav` audio files into:

- 🔴 **Dangerous Scream**  
- 🟡 **Normal Conversation**  
- 🟢 **Happy Scream**

Built using Python, Keras, Librosa, and Flask.

---

## 🚀 Features

✅ Upload `.wav` audio via web UI  
✅ MFCC feature extraction using Librosa  
✅ Trained deep learning models (CNN / LSTM)  
✅ Clean prediction display in browser  
✅ Works fully offline after setup

---

## 🧠 Models Used

- 🧪 MFCC feature extraction from raw audio  
- 🧠 Trained **CNN model** on extracted features  
- 🧠 (Optional) LSTM model (can be swapped in)  
- 🧠 (Advanced) Experiments with SVM, KNN possible

---

## 🗂 Folder Structure

```
HumanScreamDetection/
├── app.py                   # Flask backend
├── cnn_model.h5             # Trained model
├── human_scream_detection.ipynb  # Colab training notebook
├── requirements.txt         # Required packages
├── README.md                # This file
├── .gitignore
├── templates/               # HTML frontend (index.html)
├── static/                  # Optional CSS/audio files
├── model/                   # (Optional) model weights folder
└── utils/                   # Utility scripts (if any)
```

---

## 📦 How to Run the Project

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/HumanScreamDetection.git
cd HumanScreamDetection
```

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Run the Flask App

```bash
python app.py
```

Now visit `http://127.0.0.1:5000/` in your browser to upload and test audio!

---

## 🎯 Dataset Structure (from Google Drive)

```
MiniProject/
├── Scream/       
  ├── HappyScream/     # Happy Screams
  ├── DangerousScream  # Dangerous Screams 
├── NonScream          # Normal Conversation
```

## 💡 Future Improvements

- Real-time mic recording and prediction  
- Deploy to cloud (Render / Streamlit)  
- Add scream severity confidence score  
- Add mobile app integration

---

