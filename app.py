from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

img_size = 50

def pred(img):
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img).astype('float32') / 255.0
    pred = np.argmax(model.predict(np.array([img])))
    if pred == 0:
        return "USD"
    else:
        return "Euro"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    prediction = pred(img)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
