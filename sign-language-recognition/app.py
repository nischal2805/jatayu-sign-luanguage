from flask import Flask, request, jsonify
import cv2
import numpy as np
from src.config import Config
from src.models.classifier import SignLanguageClassifier

app = Flask(__name__)
model = SignLanguageClassifier()
model.load_model(Config.MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
    prediction = model.predict(img)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)