import json
import mlflow
import logging
import os
import sys
from src.logger import logging
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("")
# if not dagshub_token:
#     raise EnvironmentError(" environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "rnjt80"
# repo_name = "credit_card_fraud_detection"
# Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# # -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri("https://dagshub.com/rnjt80/credit_card_fraud_detection.mlflow")
dagshub.init(repo_owner='rnjt80', repo_name='credit_card_fraud_detection', mlflow=True)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a json file."""
    try: 
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded successfully.')
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("UNexpected error occured while loading the model info: %s", e)
        raise

def register_model_and_transformer(model_name: str, model_info: dict, transformer_name: str, transformer_path: str):
    """Register the model and power transformer to the MLflow model registerry."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Register the model
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(model_uri)
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "staging".
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging"
        )
        logging.info(f"Model {model_name} version {model_version.version} registered and transitioned to Staging.")

        # Log the PowerTransormer as an artifact.
        mlflow.log_artifact(transformer_path, artifact_path="preprocessing")

        # Register PowerTransformer in MLflow model registery.
        transformer_uri = f"runs:/{model_info['run_id']}/preprocessing/{os.path.basename(transformer_path)}"
        transformer_version  = mlflow.register_model(transformer_uri, transformer_name)

        # Transition the PowerTransformer to staging
        client.transition_model_version_stage(
            name = transformer_name,
            version = transformer_version.version,
            stage = "Staging"
        )
        logging.info(f'PowerTransformer {transformer_name} version {transformer_version.version} registered and transitioned to Staging')
    except Exception as e:
        logging.error('Error occured during model and transformation registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "fraud_detection_model"
        transformer_name = "power_transformer"
        transformer_path = "models/power_transformer.pkl"

        register_model_and_transformer(model_name, model_info, transformer_name, transformer_path)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        raise

if __name__ == '__main__':
    main()