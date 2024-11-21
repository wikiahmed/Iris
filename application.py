from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model/iris_model.pkl", "rb") as file:
    model, labels = pickle.load(file)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
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
    species_image = species_images[species_name.lower()]

    return render_template(
        'result.html', species_name=species_name, species_image=species_image
    )

if __name__ == '__main__':
    app.run(debug=True)
