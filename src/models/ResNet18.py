# ResNet model for transfer learning
from torchvision import models
from torch import nn

class ResNetClassifier:
    def __init__(self, num_classes=2, device='cuda', freeze_backbone=True):
        self.device = device
        self.model = models.resnet18(pretrained=True)

        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.model = self.model.to(self.device)

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def freeze_backbone(self):
        """Freeze all layers except final layer."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_model(self):
        """Return the model object for training."""
        return self.model