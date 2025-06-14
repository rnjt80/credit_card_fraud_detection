import unittest
import mlflow
import os
import mlflow.client
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestCreditFraudModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup dagshub creds for mlflow
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment varisable not set.")
        
        os.environ["MLFLOW_TRACKING_USER"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "rnjt80"
        repo_name = "credit_card_fraud_detection"

        # Setup mlflow tracking uri.
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the latest version of the fraud detection model from MLflow model registry.
        cls.model_name = "fraud_detection_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the powertransformer used to preprocess the data
        cls.transformer = pickle.load(open('models/power_transformer.pkl', 'rb'))

        # Load the test Datset
        cls.test_data = pd.read_csv('data/processed/test_final.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version =client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        """Ensure the Model loads without errors."""
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        """Validate the model's input signature using transformed test data."""
        sample_input = self.test_data.iloc[:1, :-1] #Take first row, exclude lable
        transformed_input = self.transformer.transform(sample_input)
        input_df = pd.DataFrame(transformed_input)

        # Make a prediction
        prediction = self.model.predict(input_df)

        #Ensure input dimentions are correct
        self.assertEqual(input_df.shape[1], self.transformer.n_features_in_)

        # Ensure output dimentions  matched expected classification output.
        self.assertEqual(len(prediction), 1) # Single prediction for one row,

    def test_model_performance(self):
        """ Evaluate model performance against the test dataset."""
        # Extract features and and lables
        X_test = self.test_data.iloc[:, :-1] # Exclude the target column
        y_test = self.test_data.iloc[:, -1] # target column

        # Transform features.
        X_test_transformed = self.transformer.transform(X_test)

        # Predict using model.
        y_pred = self.model.predict(X_test_transformed)

        # Compute performance metrcis.
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Define minimum performance thresholds
        expected_accuracy = 0.50
        expected_precision = 0.50
        expected_recall = 0.50
        expected_f1 = 0.50

        # Assert that model meets performance requirements
        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy should be atleast {expected_accuracy}.")
        self.assertGreaterEqual(precision, expected_precision, f"Precision should be atlest {expected_precision}.")
        self.assertGreaterEqual(recall, expected_recall, f"Recall should be at least {expected_recall}.")
        self.assertGreaterEqual(f1, expected_f1, f"F1 should be at least {expected_f1}.")

if __name__ == '__main__':
    unittest.main()