from flask import Flask, render_template, url_for, request
from flask_material import Material
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv(r"C:\\Users\\Waqas Ahmed\\Downloads\\Programs\\Datasets\\iris.csv")
    return render_template("preview.html", df_view=df)

@app.route('/', methods=["POST"])
def analyze():
    if request.method == 'POST':
        petal_length = request.form['petal_length']
        sepal_length = request.form['sepal_length']
        petal_width = request.form ['petal_width' ]
        sepal_width = request.form ['sepal_width' ]
        model_choice = request.form['model_choice']

        # Clean the data by convert from unicode to float
        sample_data = [sepal_length, sepal_width, petal_length, petal_width]
        clean_data = [float(i) for i in sample_data]

        # Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1, -1)

        # Reloading the Model
        if model_choice == 'logitmodel':
            logit_model = joblib.load(r"C:\\Users\\Waqas Ahmed\\Downloads\\Programs\\model\\CLF.pkl")
            result_prediction = logit_model.predict(ex1)
        else:
            # You can load other models based on user choice here if needed
            result_prediction = None
        
        # Mapping predictions to image filenames
        image_map = {
            'setosa': 'static/images/iris_setosa.jpg',
            'versicolor': 'static/images/iris_versicolor.jpg',
            'virginica': 'static/images/iris_virginica.jpg'
        }
        
        predicted_species = result_prediction[0] if result_prediction is not None else None
        image_path = image_map.get(predicted_species, None)  # Default to None if no match
        
    return render_template(
        'index.html', 
        petal_width=petal_width,
        sepal_width=sepal_width,
        sepal_length=sepal_length,
        petal_length=petal_length,
        clean_data=clean_data,
        result_prediction=result_prediction,
        model_selected=model_choice,
        image_path=image_path  # Pass image path to template
    )


if __name__ == '__main__':
    app.run(debug=True)
