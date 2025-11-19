import os
from utils.ResNet18_transforms import train_transform, val_transform
from dotenv import load_dotenv
from torchvision import datasets
# Loading dataset paths
load_dotenv()
train_dir = os.getenv('ORG_TRAIN_DIR')
test_dir  = os.getenv('ORG_TEST_DIR')

# Loading in datasets
train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
validate_data = datasets.ImageFolder(root=train_dir, transform=val_transform)
test_data  = datasets.ImageFolder(root=test_dir, transform=val_transform)