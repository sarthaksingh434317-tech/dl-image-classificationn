from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("cat_dog_model.keras")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize((150, 150))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0][0]

    if prediction > 0.5:
        result = "Dog 🐶"
    else:
        result = "Cat 🐱"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)