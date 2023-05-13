import json
import copy

import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler
import albumentations as A
from captum import GuidedGradCam
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

def test_model(model, dataloader, n_classes=3):
    """Test the model on unseen data."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    n_samples = len(dataloader.dataset)

    # compute the following stats
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
    plt.imshow(cm.to("cpu").numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(ds_labels))
    plt.xticks(tick_marks, list(ds_labels.keys()), rotation=45)
    plt.yticks(tick_marks, list(ds_labels.keys()))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./img/cm.pdf")


def plot_preds(preds_all, labels_all, ds_labels, dataloder, rows=3, cols=3):
    """Plot the worst *k* predictions"""
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
    plt.savefig("./fails.pdf")
    plt.show()


def plot_lr_vs_loss():
    """Plot learning rate vs loss, used to determine decent learning rate."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model
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
        # sample a random batch of data
        img_b, label_b = next(iter(dataloaders["train"]))
        img_b, label_b = img_b.to(device), label_b.to(device)

        # compute the loss
        output = model(img_b)
        loss = criterion(output, label_b)
        # print(loss.item())

        # do the backward-pass
        optimizer.zero_grad()
        loss.backward()

        # change the learning rate of the optimizer and update the model
        for param_group in optimizer.param_groups: param_group['lr'] = lrs[k].item()
        optimizer.step()

        # store the current learning rate and losses
        lri.append(lre[k].item())
        losses.append(loss.item())

    plt.plot(lre, losses)
    plt.title("Learnig rate vs Loss.")
    plt.xlabel("learning rate exponent")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("./img/lr_vs_loss.pdf")


def plot_guided_grad_cam(model, dataloder1, dataloder2, rows=3, cols=3):
    """Compute pixel importance (i.e grad of loss w.r.t each pixel) for `k` random images."""
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
            axes[i, j].imshow(attribution[i * rows + j] * 100)
            axes[i, j].imshow(imgs_real[i * rows + j], alpha=0.15)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.savefig("./grad_cam.pdf")
    plt.show()
    

if __name__ == "__main__":
    # test on the validation set
    model = load_resnet50(n_classes=len(ds_labels))
    state_dict = torch.load(f"/kaggle/input/resnet50/resnet50_ds.pt")
    model.load_state_dict(state_dict)

    plt.plot(list(range(N_EPOCHS)), losses["train"], label="train")
    plt.plot(list(range(N_EPOCHS)), losses["val"], label="val")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./img/loss_vs_epochs.pdf")

    plt.plot(list(range(N_EPOCHS)), f1s["train"], label="train")
    plt.plot(list(range(N_EPOCHS)), f1s["val"], label="val")
    plt.title("F1-Score vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./img/f1_vs_epochs.pdf")

    model = load_resnet50(n_classes=N_CLASSES)
    preds_all, label_all = test_model(model, dataloaders["test"])
    
    test_dataloder = get_dataloader(
        data_dir="../../data",
        annot_files=annot_files[-1],
        transform=get_basic_transform(),
        batch_size=BATCH_SIZE
    )

    plot_preds(
        preds_all, label_all, ds_labels, test_dataloader, rows=3, cols=3)


    test_dataloader1 = get_testdataloader(batch_size=32, transform=get_test_transform())
    test_dataloader2 = get_testdataloader(batch_size=32, transform=A.Compose([A.Resize(width=224, height=224)]))
    plot_guided_grad_cam(model, test_dataloader1, test_dataloader2, rows=3, cols=3)
    