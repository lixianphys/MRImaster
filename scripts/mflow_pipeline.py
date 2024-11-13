import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(project_root)
from metaflow import FlowSpec, step, Parameter
import mlflow
import torch
from src.inference.predict import load_cnn_model, load_unet3d_model, cnn_inference, unet3d_inference
from train_model import train_model  # Replace with your actual training function

class MLFlowPipeline(FlowSpec):

    # Define parameters for training and model settings
    model_type = Parameter('model_type', help="Model type to use for training/prediction ('cnn' or 'unet3d')", default="cnn")
    use_mlflow = Parameter('use_mlflow', help="Enable MLflow logging", default=True)
    epochs = Parameter('epochs', help="Number of epochs for training", default=10)
    batch_size = Parameter('batch_size', help="Batch size for training", default=32)
    learning_rate = Parameter('learning_rate', help="Learning rate", default=0.001)
    prediction_data_path = Parameter('prediction_data_path', help="Path to input data for prediction", default=None)

    @step
    def start(self):
        """Starting point of the pipeline"""
        print("Starting the MLFlow Pipeline.")
        self.next(self.train_model)

    @step
    def train(self):
        """Training step"""
        print(f"Training {self.model_type} model.")
        
        if self.use_mlflow:
            mlflow.set_experiment("MLFlow_Metaflow_Experiment")
            with mlflow.start_run():
                # Log parameters to MLflow
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("batch_size", self.batch_size)
                mlflow.log_param("learning_rate", self.learning_rate)
                
                # Call your training function, passing parameters as needed
                self.model_path = train_model(self.model_type, self.epochs, self.batch_size, self.learning_rate)

                # Log model as artifact
                mlflow.log_artifact(self.model_path)
        else:
            # Train without MLflow
            self.model_path = train_model(self.model_type, self.epochs, self.batch_size, self.learning_rate)

        self.next(self.predict)

    @step
    def predict(self):
        """Prediction step"""
        print(f"Running prediction with {self.model_type} model on data: {self.prediction_data_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "cnn":
            model = load_cnn_model(self.model_path, device)
            # Run prediction on data_path
            result = cnn_inference(model, self.prediction_data_path)
            print(f"Prediction result: {result}")

        elif self.model_type == "unet3d":
            model = load_unet3d_model(self.model_path, device)
            result = unet3d_inference(model, self.prediction_data_path)
            print("3D UNet prediction completed.")
            
        self.next(self.end)

    @step
    def end(self):
        """End step"""
        print("MLFlow Pipeline completed.")

if __name__ == "__main__":
    MLFlowPipeline()
