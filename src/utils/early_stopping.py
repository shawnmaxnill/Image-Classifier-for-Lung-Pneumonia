# Thanks to Jeff Heaton himself for providing early stopping script
# Available at https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_04_1_kfold.ipynb

import copy
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience # How many epochs to wait before stopping
        self.min_delta = min_delta # Change in loss to be counted as improvement
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):

        # Assign current best loss
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict()) # and saving best model

        # Update and saving lastest model weights if better is found 
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}\nSaving weights"
            torch.save(self.best_model, "weights.pt")
        
        # Continue normally until it reaches epoch 5
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights: # If there's a saved weight, reload it back in
                    model.load_state_dict(self.best_model)
                return True
        return False
