import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model
with open("model/iris_model.pkl", "rb") as file:
    model, labels = pickle.load(file)

# Title of the web app
st.title("🌺 Welcome to the Iris Predictor 🌼")

# Input fields for the flower dimensions
st.subheader("Enter the flower's dimensions below to discover its species!")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Button to trigger prediction
if st.button("🌟 Predict Now 🌟"):
    # Prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species_name = labels[prediction]

    # Mapping species to images
    species_images = {
        "setosa": "setosa.jpg",
        "versicolor": "versicolor.jpg",
        "virginica": "virginica.jpg"
    }

    # Display the result
    st.subheader("✨ Prediction Result ✨")
    st.write(f"The predicted species is: **{species_name}**")
    
    # Display corresponding image
    species_image = species_images[species_name.lower()]
    image = Image.open(f"images/{species_image}")
    st.image(image, caption=species_name, use_column_width=True)
    
    # Link to retry
    if st.button("🔙 Try Again"):
        st.experimental_rerun()
