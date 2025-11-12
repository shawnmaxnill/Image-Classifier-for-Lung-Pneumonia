from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn

# Defining model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    # Model flow
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Loading in trained model, weights and switching to eval mode
model = NeuralNetwork()
model.load_state_dict(torch.load("weights_v1.pt", map_location="cpu"))
model.eval()

# Image preprocessing
# Change to grey, resize and convert to tensor
preprocessing_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Initialize API
app = FastAPI(title="Image Classification")

@app.get()
def root():
    return {"You are now in localhost runnning an Image Classifier"}

# Creating endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_img = Image.open(file.file).convert("L")
    transformed_img = preprocessing_transform(input_img).unsqueeze(0)

    # Execute prediction
    with torch.no_grad():
        output = model(transformed_img)
        pred_result = torch.argmax(output, dim = 1).item()

        # Return an output
        return JSONResponse({"predicted_result": int(pred_result)})