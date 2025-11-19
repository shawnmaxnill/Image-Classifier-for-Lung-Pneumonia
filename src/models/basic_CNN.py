import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 256x256x1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # After 5 pools: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32x32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 16x16
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 8x8
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    
def get_basic_model():
    return NeuralNetwork()