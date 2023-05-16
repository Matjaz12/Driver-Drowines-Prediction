import torch
import torch.nn as nn
from torchvision import models


def swap_head(model, n_classes):
    """Swap the final fully connected layer."""
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_resnet50(n_classes):
    """Load a ResNet50 model with a swapped head."""
    resnet50 = models.resnet50(weights="IMAGENET1K_V2")
    return swap_head(resnet50, n_classes)
