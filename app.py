from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved models
log_model = joblib.load('notebooks/log_model.pkl')
tree_model = joblib.load('notebooks/tree_model.pkl')
forest_model = joblib.load('notebooks/forest_model.pkl')
xgb_model = joblib.load('notebooks/xgb_model.pkl')

# Endpoint for predictions using Logistic Regression
@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.json  # Get JSON input data
    input_data = np.array(data['input']).reshape(1, -1)  # Convert to numpy array
    prediction = log_model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

# Endpoint for predictions using Decision Tree
@app.route('/predict/tree', methods=['POST'])
def predict_tree():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = tree_model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

# Endpoint for predictions using Random Forest
@app.route('/predict/forest', methods=['POST'])
def predict_forest():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = forest_model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

# Endpoint for predictions using XGBoost
@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = xgb_model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
