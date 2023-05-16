import time
import copy
import json
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from load_data import (get_basic_transform, get_dataloaders,
                       get_test_transform, get_train_transform,
                       load_driver_data)
from resnet50 import load_resnet50
from resmasknet import load_resmasknet
from torch.optim import lr_scheduler
from torchmetrics.classification import (Accuracy, ConfusionMatrix, F1Score,
                                         Precision, Recall)
from torchvision import models


def train(dataloaders, model, criterion, optimizer, scheduler, n_epochs, n_classes, model_path):
    """Train and save the model with the highest f1-score on the validation set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

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
    losses, f1s  = {"train": [], "val": []}, {"train": [], "val": []}
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs - 1}', flush=True)
        start = time.time()
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

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
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:4f}', flush=True)

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1_score:
                best_f1_score = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
        print(f"time[s]: {time.time() - start}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model, losses, f1s


def save_history(losses, f1s, results_dir):
    """Save training history."""
    print("Saving training history...", flush=True)
    np.save(open(f"{results_dir}/losses_train.npy", "wb"), np.array(losses["train"]))
    np.save(open(f"{results_dir}/losses_val.npy", "wb"), np.array(losses["val"]))
    np.save(open(f"{results_dir}/f1s_train.npy", "wb"), np.array(f1s["train"]))
    np.save(open(f"{results_dir}/f1s_val.npy", "wb"), np.array(f1s["val"]))


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="resnet50")
    args = parser.parse_args()

    # load hyper-parameters and dataset
    params = json.load(open(os.path.join(os.getcwd(), "hyper_params.json")))
    dataloaders, ds_labels, labels_ds = load_driver_data(params)
    
    # load model
    if args.model == "resnet50":
        model = load_resnet50(n_classes=params["n_classes"])
    elif args.model == "resmasknet":
        model = load_resmasknet(n_classes=params["n_classes"])
    else:
        raise Exception(f"Model: {args.model} not supported!")

    # train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=0.1)
    model, losses, f1s = train(
        dataloaders, model, criterion, optimizer,
        scheduler, params["n_epochs"], 
        params["n_classes"], model_path=f"./{args.model}/{args.model}_ds.pt"
    )
    # save training history
    save_history(losses, f1s, results_dir=f"./{args.model}/results")
