from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model.h5', compile=True)

img_size = 50
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def pred(img):
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img).astype('float32') / 255.0
    out = model.predict(np.array([img]))
    pred = np.argmax(out)
    return f"USDの確率 {out[0][0]} euroの確率 {out[0][1]}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename('usd01.jpg')  # 固定されたファイル名
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        prediction = pred(img)
        return jsonify({'prediction': prediction, 'image_url': f'/uploads/{filename}'})
    return jsonify({'error': 'Invalid file format'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
