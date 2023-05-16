import cv2
import numpy as np
import torch
import torch.nn as nn
from rmn import get_emo_model
from torchvision.transforms import transforms


def swap_head(model, n_classes):
    """Swap the final fully connected layer."""
    model.fc[1] = nn.Linear(model.fc[1].in_features, n_classes)
    return model


def load_resmasknet(n_classes=3):
    """Load the ResidualMaskingNetwork (https://github.com/phamquiluan/ResidualMaskingNetwork)"""
    return swap_head(get_emo_model(), n_classes)
