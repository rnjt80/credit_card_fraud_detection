import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import dagshub
from flask import Flask, render_template, request
from src.logger import logging
# #------------------------------------------------
# Below code is for production use
# #------------------------------------------------
# Set up Dagshub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_TOKEN environment variable not set.")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "rnjt80"
# repo_name = "credit_card_fraud_detection"
# # Set up MLflow tracking URI.
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# #------------------------------------------------

#------------------------------------------------
# Below code is for local use
#------------------------------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/rnjt80/credit_card_fraud_detection.mlflow"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
dagshub.init(repo_owner='rnjt80', repo_name='credit_card_fraud_detection', mlflow=True)


#----------------------------------------------
# Configuration
#----------------------------------------------
MODEL_NAME = "fraud_detection_model"
PREPROCESSOR_PATH = "models/power_transformer.pkl"

# Initialize flask app
app = Flask(__name__)

#------------------------------------
# Load model and preprocessor
#------------------------------------
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version)).version
        return latest_version
    except Exception as e:
        logging.error('Error fetching model version: %s', e)
        return None

def load_model(model_name):
    """Load the latest mdoel from mlflow."""
    model_version = get_latest_model_version(model_name)
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
        logging.info(f"Loading model from: {model_uri}")
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    return None

def load_preprocessor(preprocessor_path):
    """Load the PowerTransformer from file."""
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading Powertransformer: {e}")
        return None
    
# load ML components
model = load_model(MODEL_NAME)
power_transformer = load_preprocessor(PREPROCESSOR_PATH)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amout"]

#---------------------------------
# Helper functions
#---------------------------------
def preprocess_input(data):
    """Preprocess the user input before prediction"""
    try:
        input_array = np.array(data).reshape(1, -1) # Ensure correct shape
        transformed_input = power_transformer.transform(input_array) # apply transformation
        return transformed_input
    except Exception as e:
        logging.error(f"Preprocessing Error: {e}")
        return None
    
# --------------------------------
# Routes
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_values = [""] * len(FEATURE_NAMES) # Ensure placeholder for form
    
    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()

        if csv_input: 
            try:
                values = list(map(float, csv_input.split(",")))

                if len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"Error: Expected {len(FEATURE_NAMES)} values, but got {len(values)}")
                
                input_values = values
                transformed_features = preprocess_input(input_values)
                
                if transformed_features is not None and model:
                    result = model.predict(transformed_features)
                    prediction =  "Fraud" if result[0] == 1 else "Non-Fraud"
                else:
                    prediction = "Error: Model or transformer not loaded properly."
            except ValueError as ve:
                prediction = f"Input error: {ve}"
            except Exception as e:
                prediction = f"Processing error: {e}"
    return render_template("index.html", result = prediction, csv_input=",".join(map(str, input_values)))

@app.route("/predict", methods=["POST"])
def predict():
    csv_input = request.form.get("csv_input", "").strip()
    if not csv_input:
        return "Error: no input provided"
    
    try:
        values = list(map(float, csv_input.split(",")))
        if len(values) != len(FEATURE_NAMES):
            return f"Error:Expected {len(FEATURE_NAMES)} values, but got {len(values)}."
        
        transformed_features = preprocess_input(values)
        if transformed_features is not None and model:
            result = model.predict(transformed_features)
            return "Fraud" if result[0] == 1 else "Non-Fraud"
        return "Error: Model or transformer not loaded properly."
    except Exception as e:
        return f"Error processing input: {e}"
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)