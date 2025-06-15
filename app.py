import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import os

# File paths
MODEL_PATH = 'natural_gas_demand_xgb_model.pkl'
SCALER_PATH = 'scaler.save'
FEATURE_NAMES_PATH = 'feature_names.txt'
LAST_ACTUALS_PATH = 'last_actuals.csv'

target_col = 'India total Consumption of Natural Gas (in BCM)'
# Load feature names
with open(FEATURE_NAMES_PATH, 'r') as f:
    FEATURES = f.read().splitlines()

# Load model, scaler, last actuals
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    last_actuals = pd.read_csv(LAST_ACTUALS_PATH)
    return model, scaler, last_actuals

model, scaler, last_actuals = load_model()

app = Flask(__name__)

@app.route('/')
def home():
    return 'Natural Gas Demand Forecast API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting a dict with keys matching FEATURES
    X_input = pd.DataFrame([data])
    # Feature engineering: cyclical month
    if 'Month' in X_input:
        X_input['Month'] = pd.to_datetime(X_input['Month'])
        X_input['month_sin'] = np.sin(2 * np.pi * X_input['Month'].dt.month / 12)
        X_input['month_cos'] = np.cos(2 * np.pi * X_input['Month'].dt.month / 12)
    # Fill missing engineered features with last_actuals if needed
    for col in FEATURES:
        if col not in X_input:
            X_input[col] = last_actuals[col].values[0]
    X_input = X_input[FEATURES]
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    return jsonify({'prediction': float(pred[0])})

@app.route('/retrain', methods=['POST'])
def retrain():
    # Optionally, allow retraining with new data
    # Not implemented for brevity
    return jsonify({'status': 'Retrain endpoint not implemented.'})

if __name__ == '__main__':
    app.run(debug=True)
