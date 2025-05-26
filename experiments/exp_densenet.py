import mlflow
import mlflow.pytorch
import pandas as pd
import sys
import os
from pathlib import Path


# إضافة مسار المشروع إلى نظام المسارات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.densenet_model import DenseNetModel
from my_trainer.trainer import PneumoniaTrainer

if __name__ == "__main__":
    # إعدادات التجربة
    EXPERIMENT_NAME = "chest_xray_densenet_experiment"
    MODEL_NAME = "DenseNet121"
    DATA_MAP_PATH = os.path.normpath("load_data/resized/data_map_fixed.csv")
    
    # بدء تسجيل التجربة
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # 1. تهيئة المدرب
        trainer = PneumoniaTrainer(
            model=DenseNetModel(pretrained=True),  # استخدام DenseNet121 بدلاً من ResNet
            data_map_path=DATA_MAP_PATH,
            batch_size=32,
            image_size=(224, 224),  # DenseNet يتوقع 224x224 كمدخل
            experiment_name=f"{MODEL_NAME}_experiment_v1.0"
        )

        # 2. تحميل البيانات
        print("Loading data...")
        trainer.load_data()
        
        # 3. بدء التدريب
        print("Starting training...")
        trainer.train(epochs=5)  # يمكنك تعديل عدد العصور

    

        print("Training completed and logged to MLflow!")
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("batch_size", trainer.batch_size)
        mlflow.log_param("image_size", trainer.image_size)