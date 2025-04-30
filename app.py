import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and their corresponding field names
        form_data = request.form
        fields = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'year_built']
        data = []
        
        # Convert inputs to floats and validate
        for field in fields:
            value = form_data.get(field, '').strip()
            try:
                data.append(float(value) if value else 0.0)  # Default to 0 if empty
            except ValueError:
                raise ValueError(f"Invalid input for '{field.upper()}': '{value}' is not a valid number")

        if len(data) != regmodel.n_features_in_:
            raise ValueError(f"Expected {regmodel.n_features_in_} features, got {len(data)}")

        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]

        # Convert to INR
        prediction_in_usd = output * 1000
        conversion_rate = 83
        prediction_in_inr = prediction_in_usd * conversion_rate

        # Format in lakh or crore
        if prediction_in_inr >= 10000000:
            prediction_in_crore = prediction_in_inr / 10000000
            formatted_prediction = f"₹{prediction_in_crore:.2f} Crore"
        else:
            prediction_in_lakh = prediction_in_inr / 100000
            formatted_prediction = f"₹{prediction_in_lakh:.2f} Lakh"

        return render_template("home.html", prediction_text=f"THE HOUSE PRICE PREDICTION IS: {formatted_prediction}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000)
