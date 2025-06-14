import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tacking.
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environmnet variable not set.")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "rnjt80"
    repo_name = "credit_card_fraud_detection"

    # set mlflow tracking uri
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = mlflow.MlflowClient()

    model_name = "fraud_detection_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current version in production
    prod_version = client.get_latest_versions(model_name, stage=["Production"])
    for version in prod_version:
        client.transition_model_version_stage(
            name = model_name,
            version = version.version,
            stage = "Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name = model_name,
        version = latest_version_staging,
        stage = "Production"
    )

    print('Model version {latest_version_staging} promoted to Production.')

if __name__ == '__main__':
    promote_model()