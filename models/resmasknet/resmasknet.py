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


if __name__ == "__main__":
    # Inference example
    pass
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_resmasknet()
    model = model.to(device)
    img = np.random.randint(2**8, size=(224, 224, 3), dtype=np.uint8)
    transform = transforms.Compose(
        transforms=[transforms.ToPILImage(), transforms.ToTensor()])
    img = transform(img).to(device)
    img = torch.unsqueeze(img, dim=0)
    pred = model(img)
    print(torch.softmax(pred, dim=1))
    """