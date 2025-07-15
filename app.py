# app.py
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from utils.audio_utils import extract_features
import numpy as np

UPLOAD_FOLDER = 'static/uploaded_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/scream_detection_cnn.keras', compile=False)
labels = {0: "Normal Conversation", 1: "Happy Scream", 2: "Dangerous Scream"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    audio = request.files['audio']
    filename = secure_filename(audio.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.save(filepath)

    features = extract_features(filepath)
    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        'label': labels[predicted_class],
        'confidence': f'{confidence * 100:.2f}%',
        'file_path': filepath
    })

if __name__ == '__main__':
    app.run(debug=True)
