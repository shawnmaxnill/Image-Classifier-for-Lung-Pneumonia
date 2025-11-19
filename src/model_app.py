from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from models.ResNet18 import ResNetClassifier
from utils.ResNet18_transforms import val_transform
from utils.test import test
from torchvision import datasets
from torch.utils.data import DataLoader

    
# Loading in trained model, weights and switching to eval mode
model = ResNetClassifier()
model.load_state_dict(torch.load("weights_v1.pt", map_location="cpu"))
model.eval()

# Initialize API
app = FastAPI(title="Image Classification")

@app.get()
def root():
    return {"You are now in localhost runnning an Image Classifier"}

# Creating endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_img = Image.open(file.file).convert("RGB")
    transformed_img = val_transform(input_img).unsqueeze(0)

    # Execute prediction
    with torch.no_grad():
        output = model(transformed_img)
        pred_result = torch.argmax(output, 1).item()  
        probabilities = torch.softmax(output, 1)
        confidence = probabilities[0][pred_result].item()

    # Setting up labels
    class_names = ["Normal", "Pneumonia"]
    predicted_label = class_names[pred_result]

    # Return result
    return JSONResponse({
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4)
    })