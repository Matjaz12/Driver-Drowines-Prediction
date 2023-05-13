import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler
import albumentations as A
import torch
import copy
import cv2
from torchmetrics.classification import (
    Accuracy, F1Score, Precision, Recall, ConfusionMatrix
)
from load_data import get_dataloaders


def swap_head(model, n_classes):
    """Swap the final fully connected layer."""
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_resnet50(n_classes):
    """Load a ResNet50 model with a swapped head."""
    resnet50 = models.resnet50(weights="IMAGENET1K_V2")
    return swap_head(resnet50, n_classes)


def train(dataloaders, model, criterion, optimizer, scheduler, n_epochs, n_classes=3):
    """Train the model (ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    # define metrics to keep track of
    # macro F1 score provides a better measure of the model's overall performance on all classes.
    acc = Accuracy(task="multiclass", num_classes=3).to(device)
    f1 = F1Score(task="multiclass", average="macro",
                    num_classes=n_classes).to(device)
    precision = Precision(task="multiclass", average='macro',
                            num_classes=n_classes).to(device)
    recall = Recall(task="multiclass", average='macro',
                    num_classes=n_classes).to(device)

    # we optimize for f1 score
    best_f1_score = 0.0

    losses = {"train": [], "val": []}
    f1s = {"train": [], "val": []}

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs - 1}')

        for phase in ["train", "val"]:
            if phase == "train": model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_f1 = 0.0
            running_precision = 0.0
            running_recall = 0.0

            for idx, (img, label) in enumerate(dataloaders[phase]):
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = model(img)
                    loss = criterion(output, label)
                    _, preds = torch.max(output, dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(preds == label.data)
                running_f1 += f1(preds, label)
                running_precision += precision(preds, label)
                running_recall += recall(preds, label)

            if phase == "train":
                scheduler.step()

            n_samples = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            epoch_f1 = running_f1 / (idx + 1)
            epoch_precision = running_precision / (idx + 1)
            epoch_recall = running_recall / (idx + 1)

            losses[phase].append(epoch_loss)
            f1s[phase].append(epoch_f1.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:4f}')

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1_score:
                best_f1_score = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses, f1s


def get_train_transform():
    """Get the train transform."""
    IMAGENET_RES = (224, 224)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = A.Compose([
        A.Resize(width=IMAGENET_RES[0], height=IMAGENET_RES[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.OneOf([
        A.Rotate(limit=20, p=0.35, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.65)],
        p=0.75),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.75)
    ])
    return transform


def get_test_transform():
    """Get the test transform."""
    IMAGENET_RES = (224, 224)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = A.Compose([
        A.Resize(width=IMAGENET_RES[0], height=IMAGENET_RES[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


if __name__ == "__main__":
    annot_files = ["classification_frames/annotations_train.json",
                    "classification_frames/annotations_val.json",
                    "classification_frames/annotations_test.json"]

    dataloaders, ds_labels, labels_ds = get_dataloaders(
        data_dir="../../data",
        annot_files=annot_files,
        class_weights=[1/34792, 1/4620, 1/3970],
        transforms=[get_train_transform(), get_test_transform(), get_test_transform()],
        batch_size=32
    )

    for _, label in dataloaders["train"]:
        print(f"distribution of labels, should be somewhat uniform: ", label)
        break

    # Load the model
    model = load_resnet50(n_classes=len(ds_labels))
    pred = model(next(iter(dataloaders["train"]))[0])
    print(pred.shape)
