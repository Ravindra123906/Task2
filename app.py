from flask import Flask, render_template, request
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_model.h5')
model = load_model(model_path)

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction < 0.5:
        return "Healthy"
    else:
        return "Diseased"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            result = predict_disease(file_path)
            return render_template('index.html', result=result, img_path=file_path)
    return render_template('index.html', result=None, img_path=None)

if __name__ == '__main__':
    app.run(debug=True)
