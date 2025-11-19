# Package Imports
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Subset, WeightedRandomSampler
from torch import nn

# File Imports
from dataloader.training_dataloader import get_train_val_loaders, get_test_loader
from dataloader.loading_dataset import train_data, test_data
from models.ResNet18 import ResNetClassifier
from utils.early_stopping import EarlyStopping
from utils.train import train
from utils.test import test
from utils.metrics import print_metrics
from utils.plotting import plot_all_folds

# Function to merge both train and testing for CV
def run_sequence(train_DL, test_DL, model, loss_fn, optimizer, epochs):

    # Statistics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    early_stopping = EarlyStopping(patience=5)

    for epoch in range(epochs):

        print(f"Epoch Number {epoch + 1}")
        train_loss = train(train_DL, model, loss_fn, optimizer)
        accuracy, val_loss = test(test_DL, model, loss_fn)

        # Storing statistics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        # Early stopping
        stop = early_stopping(model, val_loss)
        print(early_stopping.status)
        if stop:
            print("Early stopping triggered!")
            break

    return history

# Device and Set Seed
torch.manual_seed(40)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Setting up Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
# Early stopping params
batch_size = 16
epochs = 10
freezed_history_lst = []
unfreezed_history_lst = []

# Main SKFold Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_data.targets)), train_data.targets)):
    # Subset Data from current SKFold
    print(f"Fold: {fold}")

    train_subset = Subset(train_data, train_idx)
    val_subset = Subset(train_data, val_idx)

    # Training Data modification via Weighted sampler
    train_targets = np.array(train_data.targets) 
    train_targets = train_targets[train_idx] # Retrieve data only in the SKFold
    class_counts = np.bincount(train_targets)
    class_weights = 1. / torch.tensor(class_counts, dtype= torch.float32)

    # Assigning weights to all labels (Weights Generation)
    sample_weights = class_weights[train_targets]

    train_sampler = WeightedRandomSampler(
        weights= sample_weights,
        num_samples= len(sample_weights),
        replacement=True
        )
    
    # Getting DataLoaders
    train_loader, val_loader = get_train_val_loaders(train_data, train_idx, val_idx, batch_size=batch_size)

    # Reinitializing model and optimizers =================================
    resnet = ResNetClassifier(freeze_backbone=True)
    optimizer = torch.optim.Adam(resnet.get_model().fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Freezing model
    print("Classifier Training (Final Layer only)")
    resnet.freeze_backbone()
    freezed_model_hist = run_sequence(train_loader, val_loader, resnet.get_model(), loss_fn, optimizer, epochs)

    # Unfreeze model
    resnet.unfreeze_backbone()
    optimizer = torch.optim.Adam(resnet.get_model().parameters(), lr=1e-4) # lesser learning rate for fine tuning

    print("Fine tuning entire network, Unfreezing model")
    resnet.unfreeze_backbone()
    unfreezed_model_hist = run_sequence(train_loader, val_loader, resnet.get_model(), loss_fn, optimizer, epochs)

    freezed_history_lst.append(freezed_model_hist)
    unfreezed_history_lst.append(unfreezed_model_hist)


# Printing summary metrics
print_metrics(unfreezed_history_lst)

# Flatten function for plotting
def flatten_loss(loss_list):
    # If elements are lists, flatten them
    if isinstance(loss_list[0], list) or isinstance(loss_list[0], tuple):
        return [x[0] for x in loss_list]
    return loss_list

plot_all_folds(epochs, unfreezed_history_lst)

# Testing model on unseen data
resnet = ResNetClassifier(num_classes=2)
model = resnet.get_model()
model.load_state_dict(torch.load("weights.pt"))


test_loader = get_test_loader(test_data, batch_size)
test(test_loader, model, loss_fn)


