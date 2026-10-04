import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐱 Cat vs Dog 🐶")

@st.cache_resource
def load():
    return tf.keras.models.load_model("cat_dog_model.keras")

model = load()
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    arr = np.expand_dims(np.array(img.resize((150, 150))) / 255.0, axis=0)
    p = model(arr, training=False).numpy()[0][0]
    st.subheader("Dog 🐶" if p > 0.5 else "Cat 🐱")