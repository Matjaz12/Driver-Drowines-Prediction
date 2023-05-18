import argparse
import copy
import json
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import GuidedGradCam
from torch.optim import lr_scheduler
from torchmetrics.classification import (Accuracy, ConfusionMatrix, F1Score,
                                         Precision, Recall)
from torchvision import models
from tqdm import tqdm

from load_data import (get_basic_transform, get_dataloader, load_driver_data, get_dataloaders,
                       get_test_transform, get_train_transform, ds_labels, label_ds, annot_files, ROOT_DIR)
from resnet50 import load_resnet50
from resmasknet import load_resmasknet



def test_model(model, dataloader, ds_labels, save_path, n_classes=3):
    """Test the model on unseen data."""
    print("testing model...", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    n_samples = len(dataloader.dataset)
    acc_fun = Accuracy(task="multiclass", num_classes=3).to(device)
    f1_fun = F1Score(task="multiclass", average="macro", num_classes=n_classes).to(device)
    precision_fun = Precision(task="multiclass", average='macro', num_classes=n_classes).to(device)
    recall_fun = Recall(task="multiclass", average='macro', num_classes=n_classes).to(device)
    acc, f1, precision, recall = 0.0, 0.0, 0.0, 0.0
    preds_all, label_all = [], []

    for img, label in tqdm(dataloader, desc="Testing the model."):
        img, label = img.to(device), label.to(device)
        preds = model(img).argmax(dim=1)        
        preds_all.append(preds)
        label_all.append(label)
    preds_all = torch.cat(preds_all, dim=0)
    label_all = torch.cat(label_all, dim=0)

    # compute over all predictions and labels
    acc = acc_fun(preds_all, label_all)
    f1 = f1_fun(preds_all, label_all)
    precision = precision_fun(preds_all, label_all)
    recall = recall_fun(preds_all, label_all)
    print(f"(test_model) acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")    

    cm_fun = ConfusionMatrix(task="multiclass", num_classes=3).to(device)
    cm = cm_fun(preds_all, label_all)
    print(cm)
    plt.figure()
    plt.imshow(cm.to("cpu").numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(ds_labels))
    plt.xticks(tick_marks, list(ds_labels.keys()), rotation=45)
    plt.yticks(tick_marks, list(ds_labels.keys()))
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    save_path = os.path.join(save_path, "cm.pdf")
    plt.savefig(save_path)
    return preds_all, label_all


def plot_preds(preds_all, label_all, ds_labels, dataloder, save_path, rows=3, cols=3):
    """Plot the worst *k* predictions"""
    print("plotting predictions...", flush=True)
    def img_tensor_to_img(img_t):
        """Convert a batch of image tensors to a batch of images."""
        return img_t.numpy().transpose(0, 2, 3, 1)
    k = rows * cols
    wrong_idx = torch.where(preds_all != label_all)[0]
    wrong_idx = wrong_idx[torch.randint(low=0, high=len(wrong_idx), size=(k, ))]

    # fetch the wrong data
    imgs = torch.stack([dataloder.dataset[idx][0] for idx in wrong_idx], dim=0)
    imgs = img_tensor_to_img(imgs.cpu())
    labels = torch.stack([dataloder.dataset[idx][1] for idx in wrong_idx], dim=0)
    preds = preds_all[wrong_idx]

    labels_ds = {v:k for k, v in ds_labels.items()}
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle('False Predictions', fontsize=16)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(imgs[i * rows + j])
            axes[i, j].set_title(
                f"label: {labels_ds[labels[i * rows + j].item()]}, pred: {labels_ds[preds[i * rows + j].item()]}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.tight_layout()
    save_path = os.path.join(save_path, "fails.pdf")
    plt.savefig(save_path)


def plot_lr_vs_loss(dataloaders, save_path):
    """Plot learning rate vs loss, used to determine decent learning rate."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_resnet50(n_classes=len(ds_labels))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # define a set of lr-values
    lre = torch.linspace(-4, -1, 1000)
    lrs = 10 ** lre
    n_iters = len(lrs)
    lri = []
    losses = []
    for k in tqdm(range(n_iters)):
        img_b, label_b = next(iter(dataloaders["train"]))
        img_b, label_b = img_b.to(device), label_b.to(device)
        output = model(img_b)
        loss = criterion(output, label_b)
        optimizer.zero_grad()
        loss.backward()
        for param_group in optimizer.param_groups: param_group['lr'] = lrs[k].item()
        optimizer.step()
        lri.append(lre[k].item())  # store the current learning rate and losses
        losses.append(loss.item())
    plt.figure()
    plt.plot(lre, losses)
    plt.title("Learnig rate vs Loss.")
    plt.xlabel("learning rate exponent")
    plt.ylabel("loss")
    plt.tight_layout()
    save_path = os.path.join(save_path, "lr_vs_loss.pdf")
    plt.savefig(save_path)


def plot_guided_grad_cam(model, dataloder1, dataloder2, save_path, rows=3, cols=3):
    """Compute pixel importance (i.e grad of loss w.r.t each pixel) for `k` random images."""
    print("plotting guided gradcam...", flush=True)
    def img_tensor_to_img(img_t):
        """Convert a batch of image tensors to a batch of images."""
        return img_t.numpy().transpose(0, 2, 3, 1)
    
    k = rows * cols
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_idxs = torch.randint(low=0, high=len(dataloder1.dataset), size=(k, ))
    imgs = torch.stack([dataloder1.dataset[idx][0] for idx in rand_idxs], dim=0).to(device)
    labels = torch.stack([dataloder1.dataset[idx][1] for idx in rand_idxs], dim=0).to(device)
    
    imgs.requires_grad=True
    guided_gc = GuidedGradCam(model, model.layer4[-1], model.relu)
    attribution = guided_gc.attribute(imgs, labels)
    
    attribution = img_tensor_to_img(attribution.detach().cpu()).max(axis=-1)
    imgs_real = torch.stack([dataloder2.dataset[idx][0] for idx in rand_idxs], dim=0).to(device)
    imgs_real = img_tensor_to_img(imgs_real.cpu())
    
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle('Guided Grad-CAM', fontsize=16)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(imgs_real[i * rows + j], alpha=0.15)
            axes[i, j].imshow(attribution[i * rows + j] * 100)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.tight_layout()
    save_path = os.path.join(save_path, "grad_cam.pdf")
    plt.savefig(save_path)
    plt.show()
    

def load_history(results_dir):
    """Load trainig history."""
    print("loading training history...", flush=True)
    losses, f1s = {}, {}
    losses["train"] = np.load(open(f"{results_dir}/losses_train.npy", "rb"))
    losses["val"] = np.load(open(f"{results_dir}/losses_val.npy", "rb"))
    f1s["train"] = np.load(open(f"{results_dir}/f1s_train.npy", "rb"))
    f1s["val"] = np.load(open(f"{results_dir}/f1s_val.npy", "rb"))
    return losses, f1s


def plot_history(losses, f1s, results_dir):
    """Plot training history."""
    print("Plotting training history...", flush=True)
    plt.figure()
    plt.plot(np.arange(len(losses["train"])), losses["train"], label="train")
    plt.plot(np.arange(len(losses["val"])), losses["val"], label="val")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_vs_epochs.pdf")

    plt.figure()
    plt.plot(np.arange(len(f1s["train"])), f1s["train"], label="train")
    plt.plot(np.arange(len(f1s["val"])), f1s["val"], label="val")
    plt.title("F1-Score vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/f1_vs_epochs.pdf")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="resnet50")
    args = parser.parse_args()

    # load model and state dict.
    print(f"loading model {args.model}...", flush=True)
    if args.model == "resnet50":
        model = load_resnet50(n_classes=3)
    elif args.model == "resmasknet":
        model = load_resmasknet(n_classes=3)
    else:
        raise Exception(f"Model: {args.model} not supported!")
    
    state_dict_path =f"./{args.model}/{args.model}_ds.pt"
    print(f"loading state dictionary from {state_dict_path}", flush=True)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()

    # load params and history
    params = json.load(open(os.path.join(os.getcwd(), "hyper_params.json")))
    losses, f1s = load_history(results_dir=f"./{args.model}/results")
    # plot_history(losses, f1s, results_dir=f"./{args.model}/results")

    # test model
    test_dataloader_t = get_dataloader(
        data_dir=ROOT_DIR,
        annot_file=annot_files[-1],
        transform=get_test_transform(),
        ds_labels = ds_labels,
        batch_size=params["batch_size"]
    )
    preds_all, label_all = test_model(
        model, test_dataloader_t, ds_labels, save_path=f"./{args.model}/results")
    
    # plot model predictions
    test_dataloader_b = get_dataloader(
        data_dir=ROOT_DIR,
        annot_file=annot_files[-1],
        transform=get_basic_transform(),
        ds_labels = {"alert": 0, "microsleep": 1, "yawning": 2},
        batch_size=params["batch_size"]
    )

    # plot_preds(preds_all, label_all, ds_labels, test_dataloader_b, save_path=f"./{args.model}/results")
    plot_guided_grad_cam(model, test_dataloader_t, test_dataloader_b,  save_path=f"./{args.model}/results")
    