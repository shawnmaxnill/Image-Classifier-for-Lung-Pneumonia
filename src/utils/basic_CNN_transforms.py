from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convert to greyscale
    transforms.Resize((256, 256)),  # resize all images standarization
    transforms.RandomHorizontalFlip(), # Flipping data
    transforms.ToTensor(),          # convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale images
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # No flipping for validation
    transforms.Normalize(mean=[0.5], std=[0.5])
])