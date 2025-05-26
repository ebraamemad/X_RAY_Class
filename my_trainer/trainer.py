# trainer/trainer.py - النسخة النهائية المُصححة

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_trainer.dataset import ChestXrayDataset, get_global_image_cache
from my_trainer.transforms import get_train_transforms, get_test_transforms
import pandas as pd
import numpy as np


class PneumoniaTrainer:
    def __init__(self, model, data_map_path, image_size=(224, 224), batch_size=32, lr=1e-4, experiment_name="Pneumonia Detection"):
        self.model = model
        self.data_map_path = data_map_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name

        self.train_transform = get_train_transforms(self.image_size)
        self.test_transform = get_test_transforms(self.image_size)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_weights = None  # هنخليها class_weights عشان انت متعود عليها

    def load_data(self):
        """تحميل البيانات مع إصلاح مسارات الصور تلقائياً"""
        # الحصول على الكاش العالمي للصور (هيحل مشكلة المسارات)
        image_cache = get_global_image_cache()
        
        df = pd.read_csv(self.data_map_path)
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        # إنشاء datasets مع الكاش المشترك
        train_dataset = ChestXrayDataset(train_df, transform=self.train_transform, image_cache=image_cache)
        val_dataset = ChestXrayDataset(val_df, transform=self.test_transform, image_cache=image_cache)
        test_dataset = ChestXrayDataset(test_df, transform=self.test_transform, image_cache=image_cache)

        labels = train_df['clas'].values
        class_counts = np.bincount([1 if label == 'PNEUMONIA' else 0 for label in labels])
        
        # حساب pos_weight (زي ما انت عامل)
        self.class_weights = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float)
        
        # حساب sample weights للـ WeightedRandomSampler
        sample_class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = sample_class_weights[[1 if label == 'PNEUMONIA' else 0 for label in labels]]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # إنشاء DataLoaders مع num_workers=0 لتجنب مشاكل Windows
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"Class counts: Normal={class_counts[0]}, Pneumonia={class_counts[1]}")
        print(f"pos_weight: {self.class_weights}")

    def evaluate(self, model, dataloader, criterion):
        """تقييم النموذج - تم إصلاح مشكلة predictions"""
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = model(inputs)  # logits
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                
                # إصلاح المشكلة: استخدام probabilities للتنبؤ
                probs = torch.sigmoid(outputs)
                preds = torch.round(probs)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # استخدام probabilities
                
        loss = running_loss / len(dataloader.dataset)
        acc = running_corrects.double() / len(dataloader.dataset)
        auc = roc_auc_score(all_labels, all_probs)
        return loss, acc, auc

    def train(self, epochs=2):
        mlflow.set_experiment(self.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow.start_run():
            
            self.model = self.model.to(self.device)
            model = self.model
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(self.device))

            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            # تسجيل parameters في mlflow
            mlflow.log_params({
                "batch_size": self.batch_size,
                "lr": self.lr,
                "image_size": self.image_size,
                "model_name": type(self.model).__name__,
                "pos_weight": self.class_weights.item()
            })

            best_val_auc = 0
            best_model_path = "resnet_model.pth"

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)  # logits
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    probs = torch.sigmoid(outputs)  # تحويل logits إلى probabilities
                    preds = torch.round(probs)                   
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
                val_loss, val_acc, val_auc = self.evaluate(self.model, self.val_loader, criterion)

                print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")

                mlflow.log_metrics({
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc.item(),
                    "val_loss": val_loss,
                    "val_acc": val_acc.item(),
                    "val_auc": val_auc
                }, step=epoch)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # Log best model
            mlflow.pytorch.log_model(model, "model")

            # Evaluate on test set
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc, test_auc = self.evaluate(model, self.test_loader, criterion)
            print(f"Test Results — Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_acc": test_acc.item(),
                "test_auc": test_auc
            })
            mlflow.log_artifact(best_model_path)
            mlflow.log_artifact(self.data_map_path)