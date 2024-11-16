import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app=application
# Load the classifier and scaler
CLF = pickle.load(open('model/CLF.pkl', 'rb'))
SS = pickle.load(open('model/SS.pkl', 'rb')) 

# Home Route
@app.route('/')
def index():
    return render_template('home.html')  # Ensuring consistency with 'home.html'

# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve form data
            sepal_length = float(request.form.get('sepal_length'))
            sepal_width  = float(request.form.get('sepal_width'))
            petal_length = float(request.form.get('petal_length'))
            petal_width  = float(request.form.get('petal_width'))

            # Transform the input data and predict
            new_data = SS.transform([[sepal_length, sepal_width, petal_length, petal_width]])
            result = CLF.predict(new_data)[0]  # Get the first prediction result

            return render_template('home.html', result=result)

        except Exception as e:
            # Handle any exception and display an error message
            return render_template('home.html', result="Error: " + str(e))

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
