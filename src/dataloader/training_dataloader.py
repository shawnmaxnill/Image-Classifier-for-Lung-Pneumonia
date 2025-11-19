from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import torch
import numpy as np

# Training and Validation Data
def get_train_val_loaders(train_data, train_idx, val_idx, batch_size=32):
    # Subset data
    train_subset = Subset(train_data, train_idx)
    val_subset = Subset(train_data, val_idx)

    # Weighted sampler
    train_targets = np.array(train_data.targets)[train_idx]
    class_counts = np.bincount(train_targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = class_weights[train_targets]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

# Testing Data
def get_test_loader(test_data, batch_size):
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return test_loader
