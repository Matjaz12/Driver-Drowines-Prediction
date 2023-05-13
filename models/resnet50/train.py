import copy
import json

import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler
from torchmetrics.classification import (
    Accuracy, F1Score, Precision,
    Recall, ConfusionMatrix
)
from load_data import (
    get_dataloaders, get_train_transform,
    get_test_transform, get_basic_transform, 
    get_dataloader
)

from resnet50 import load_resnet50


def train(dataloaders, model, criterion, optimizer, scheduler, n_epochs, n_classes=3):
    """Train the model (ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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
                torch.save(best_model_wts.state_dict(), "./resnet50_ds.pt")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses, f1s

if __name__ == "__main__":
    print("what?")
    params = json.load(open(os.path.join(os.getcwd(), "hyper_params.json")))
    print(f"params: {params}")
    annot_files = ["classification_frames/annotations_train.json",
                    "classification_frames/annotations_val.json",
                    "classification_frames/annotations_test.json"]
    dataloaders, ds_labels, labels_ds = get_dataloaders(
        data_dir="../../data",
        annot_files=annot_files,
        class_weights=[1/38967, 1/8869, 1/5495],
        transforms=[get_train_transform(), get_test_transform(), get_test_transform()],
        batch_size=params["batch_size"]
    )
    print(f"ds_labels: {ds_labels}")

    model = load_resnet50(n_classes=params["n_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    model, losses, f1s = train(
        dataloaders, model, criterion, optimizer, scheduler, n_epochs=params["n_epochs"])
    torch.save(model.state_dict(), "./resnet50_ds.pt")
