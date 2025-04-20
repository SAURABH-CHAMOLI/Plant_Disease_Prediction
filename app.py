from flask import Flask, request, render_template, url_for
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import os
import pickle
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model_path = "model.pkl"
model_url = "https://drive.google.com/uc?export=download&id=1KAh3S_RWFcoHQIm5BHSkZiCLiEpmq1L4"

# Download the model only if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        raise Exception(f"Failed to download model: HTTP {response.status_code}")

# Load model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load class indices
with open("class_indices.json") as f:
    class_indices = json.load(f)

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[str(predicted_index)]
    return predicted_class

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename != "":
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                prediction = predict_image_class(image_path)
                image_url = url_for('static', filename=f'uploads/{filename}')
    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
