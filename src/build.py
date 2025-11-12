# Imports
import os
from dotenv import load_dotenv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Retrieving paths
load_dotenv()
train_dir = os.getenv('ORG_TRAIN_DIR')
test_dir  = os.getenv('ORG_TEST_DIR')

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((255, 255)),  # resize all images
    transforms.ToTensor(),          # convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale images
])

# Loading in datasets
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data  = datasets.ImageFolder(root=test_dir, transform=transform)

# Passing data into DataLoader
# iterable, auto batching, sampling, shuffling and multiprocessing
batch_size = 32

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Creating model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
    
model = NeuralNetwork().to(device)
print(model)
    
# Training Data
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Computing prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# Testing data aka Validation
validation_loss = []

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    validation_loss.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Epoch loop
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Epoch cycle completed")

print("Generating Loss Graph...")
plot_epoch = list(range(1, epochs + 1))

plt.figure(figsize=(8,5))
plt.plot(plot_epoch, validation_loss, marker='s', label='Validation Loss')  # optional

plt.title("Validation Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(plot_epoch)
plt.legend()
plt.grid(True)
plt.savefig("validation_plot.png")
print("Graph Generation Completed")

# Saving model
save_name = "weights_v1.pt"
torch.save(model.state_dict(), save_name)
print(f"Model weights saved as {save_name}")
print("Workflow completed")

