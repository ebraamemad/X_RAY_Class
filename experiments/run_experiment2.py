import mlflow
import torch.nn as nn
import mlflow.pytorch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.base_model import PneumoniaCNN
from my_trainer.trainer import PneumoniaTrainer



if __name__ == "__main__":
    mlflow.set_experiment("Pneumonia Detection")

    with mlflow.start_run():
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("lr", 0.0001)
        mlflow.log_param("image_size", (224, 224))
        mlflow.log_param("epochs", 2)

        model = PneumoniaCNN()
        trainer = PneumoniaTrainer(
            data_map_path= os.path.normpath("load_data/resized/data_map_fixed.csv"),
            model=model,
            image_size=(224, 224),
            batch_size=64
        )

        trainer.load_data()
        trainer.train(epochs=2)
        
        
        mlflow.log_param("model2", "cnn")
        mlflow.log_param("batch_size", trainer.batch_size)
        mlflow.log_param("image_size", trainer.image_size)