import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import cv2
import json
from tqdm import tqdm


class DriverStateDataset(Dataset):
    """Driver state (classification) dataset"""

    def __init__(self, data_dir, img_paths, labels, label_dict, transform=None):
        self.data_dir = data_dir
        self.img_paths = img_paths
        self.labels = labels
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_paths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        driver_state = self.label_dict[self.labels[idx]["driver_state"]]
        img = self.transform(image=img)["image"] if self.transform else img
        img = ToTensorV2()(image=img)['image']
        driver_state = torch.tensor(driver_state)
        return img, driver_state


def get_dataloaders(
    data_dir, annot_files: list, class_weights: list,
    transforms: list = [None, None, None], batch_size: int = 32
):
    """Get train, validation and test dataloader."""

    assert len(transforms) == 3, "transforms must be a list of length 3 (train trans, val trans, test trans)"
    assert len(annot_files) == 3, "annot_files must be a list of length 3 (train annot, val annot, test annot)"

    # create datasets
    ds_labels = {"alert": 0, "microsleep": 1, "yawning": 2}
    labels_ds = {v:k for k,v in ds_labels.items()}
    datasets = []
    for idx in range(3):
        annots = json.load(open(os.path.join(data_dir, annot_files[idx])))
        img_paths = list(annots.keys())
        labels = list(annots.values())
        dataset = DriverStateDataset(
            data_dir, img_paths, labels, ds_labels, transform=transforms[idx])
        datasets.append(dataset)

    # compute sample weights
    desc = "Computing sample weights"
    sample_weights = []
    for _, label in tqdm(datasets[0], total=len(datasets[0]), desc=desc):
        sample_weights.append(class_weights[label])

    # oversample the minority classes in the **training set**
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(datasets[0]), replacement=True)
    train_dataloder = DataLoader(datasets[0], batch_size, sampler=sampler)
    val_dataloader = DataLoader(datasets[1], batch_size)
    test_dataloader = DataLoader(datasets[2], batch_size)

    return {"train": train_dataloder, "val": val_dataloader, "test": test_dataloader}, ds_labels, labels_ds
