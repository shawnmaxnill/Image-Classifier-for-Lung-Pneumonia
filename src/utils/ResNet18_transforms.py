from torchvision import transforms

# Separate transfrom settings for ResNet
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize all images standarization
    transforms.RandomHorizontalFlip(), # Flipping data
    transforms.ToTensor(),          # convert to PyTorch tensor
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # normalize grayscale images
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # No flipping for validation
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])