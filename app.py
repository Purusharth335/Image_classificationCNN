from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

# Load the trained model (saved in .h5 format)
model = tf.keras.models.load_model('model.h5', custom_objects=None, compile=True, safe_mode=True
)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# Preprocessing function
def preprocess_image(image):
    image = image.resize((32, 32))          # Resize for CIFAR-10
    image = image.convert("RGB")            # Ensure 3 channels
    image = np.array(image) / 255.0         # Normalize
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(file)
            processed = preprocess_image(image)
            pred = model.predict(processed)
            pred_class = np.argmax(pred)
            prediction = class_names[pred_class]
            print("Predicted:", prediction)  # Console debug
    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)