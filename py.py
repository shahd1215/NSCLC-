import joblib
import pandas as pd
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and feature names
combined_model = joblib.load('models/rf_model.pkl')  # Combined model for recurrence and location
scaler = joblib.load('models/min_scaler.pkl')  # Scaler for input normalization
model_features = joblib.load('models/model_features.pkl')  # List of model input features

# Encoding dictionaries for categorical variables
encoding_mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'Ethnicity': {'Caucasian': 0, 'Asian': 1, 'Native Hawaiian/Pacific Islander': 2,
                  'African-American': 3, 'Hispanic/Latino': 4},
    'Smoking status': {'Nonsmoker': 0, 'Current': 1, 'Former': 2},
    'Pathological T stage': {'Tis': 0, 'T1a': 1, 'T1b': 2, 'T2a': 3, 'T2b': 4, 'T3': 5, 'T4': 6},
    'Pathological N stage': {'N0': 0, 'N1': 1, 'N2': 2},
    'Tumor Location': {'RUL': 0, 'RML': 1, 'RLL': 2, 'LUL': 3, 'LLL': 4, 'L Lingula': 5},
    'EGFR mutation status': {'Wildtype': 0, 'Mutant': 1},
    'KRAS mutation status': {'Wildtype': 0, 'Mutant': 1},
    'ALK translocation status': {'Wildtype': 0, 'Translocated': 1},
    'Chemotherapy': {'No': 0, 'Yes': 1},
    'Radiation': {'No': 0, 'Yes': 1},
    'Survival Status': {'Dead': 0, 'Alive': 1},
    'Recurrence': {'no': 0, 'yes': 1},
    'Recurrence Location': {'No': 0, 'local': 1, 'regional': 2, 'distant': 3}
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Collect inputs from the form
        inputs = {}
        for feature in model_features:
            inputs[feature] = request.form.get(feature, None)

        # Create DataFrame for the input data
        input_data = pd.DataFrame([inputs])

        # Apply encoding for categorical fields
        for feature, mapping in encoding_mappings.items():
            if feature in input_data.columns:
                input_data[feature] = input_data[feature].map(mapping)
        input_data = input_data.reindex(columns=model_features, fill_value=0)

        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Make predictions using the combined model
        predictions = combined_model.predict_proba(input_data_scaled)
        recurrence_yes_prob = predictions[0][0]  # Probability for 'yes' class (recurrence)
        location_probs = predictions[0][1:4]  # Probabilities for location: local, regional, distant

        # Decode recurrence probability and convert to percentages
        recurrence = {
            'yes': f"{recurrence_yes_prob * 100:.1f}%",
            'no': f"{(1 - recurrence_yes_prob) * 100:.1f}%"
        }

        # Decode location probabilities if recurrence is 'yes'
        if recurrence_yes_prob > 0.5:
            total_prob = sum(location_probs)
            location_probabilities = {
                'local': f"{(location_probs[0] / total_prob) * 100:.1f}%",
                'regional': f"{(location_probs[1] / total_prob) * 100:.1f}%",
                'distant': f"{(location_probs[2] / total_prob) * 100:.1f}%"
            }
            recurrence_location = location_probabilities
        else:
            recurrence_location = "Not Applicable"

        # Return the prediction result on the result page
        return render_template('result.html', recurrence=recurrence, location=recurrence_location)

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
