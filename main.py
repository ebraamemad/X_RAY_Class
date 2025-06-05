# main.py - Simple FastAPI for Pneumonia Detection
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from PIL import Image
import io
import sys
import os

# Add your project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet_model import ResNetModel
from my_trainer.transforms import get_test_transforms

app = FastAPI()

# Global variables
model = None
transform = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
def load_model():
    """Load the trained model on startup"""
    global model, transform
    
    try:
        # Load your trained model
        model_path = "resnet_model.pth" 
        
        model = ResNetModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load transforms
        transform = get_test_transforms((224, 224))
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and the path is correct")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict pneumonia from chest X-ray image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
            prediction = "PNEUMONIA" if probability > 0.5 else "NORMAL"
        
        return {
            "prediction": prediction,
            "probability": float(probability),
            "confidence": float(probability if probability > 0.5 else 1 - probability)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=8000)
#print("device", device)