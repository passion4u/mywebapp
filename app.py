from flask import Flask, request, jsonify, render_template
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5',compile=True)

img_size = 50

def pred(img):
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img).astype('float32') / 255.0
    out = model.predict(np.array([img]))
    return f"USDの確率 {1-out} euroの確率 {out}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    prediction = pred(img)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
