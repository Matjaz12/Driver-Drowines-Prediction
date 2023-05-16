import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
from tqdm import tqdm


def extract_frames(root_dir):
    """Extract frames from videos."""
    videos_dir = os.path.join(root_dir, "videos")
    save_dir = os.path.join(root_dir, "frames") 

    for file in os.listdir(videos_dir):
        if "mp4" not in file: continue
        print(f"Extracting frames, video: {file}")
        if not os.path.exists(f"{save_dir}/{file.split('.')[-2]}"):
            os.mkdir(f"{save_dir}/{file.split('.')[-2]}")
        count = 0
        vidcap = cv2.VideoCapture(os.path.join(videos_dir, file))
        success, img = vidcap.read()
        while success:
            cv2.imwrite(f"{save_dir}/{file.split('.')[-2]}/frame{count}.jpg", img)
            success, img = vidcap.read()
            count += 1
        print(f"Frames extracted: {count}")


def annotate_seq(seq_dir, model):
    """Iterate over frames in video sequence and annotate bbox 
    and facial landmarks of the driver."""
    bbox_area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    annots = {}
    for frame in tqdm(os.listdir(seq_dir)):
        if ".jpg" not in frame:
            continue
        # load image
        img = cv2.imread(os.path.join(seq_dir, frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # make predictions
        pred = model.predict_jsons(img)
        pred = ([p for p in pred if p["score"] != -1])
        if len(pred) == 0: continue

        # keep bbox with max. area and save predictions
        max_bbox_idx = np.argmax([bbox_area(p["bbox"]) for p in pred])
        pred = pred[max_bbox_idx]
        annots[frame] = {}
        annots[frame]["bbox"] = pred["bbox"]
        annots[frame]["landmarks"] = pred["landmarks"]
    return annots


def annotate_seqs(root_dir):
    """Annotate (i.e compute face bbox and facial landmarks) sequences."""
    save_dir = os.path.join(root_dir, "frames") 
    model = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    model.eval()
    for seq in os.listdir(root_dir):
        print(f"Annotating seq: {seq}")
        annots = annotate_seq(os.path.join(save_dir, seq), model)
        with open(f"{os.path.join(save_dir, seq)}/annotations.json", "w") as file:
            json.dump(annots, file)


if __name__ == "__main__":
    ROOT_DIR = "/d/hpc/projects/FRI/DL/mm1706" # "./"
    extract_frames(ROOT_DIR)
    annotate_seqs(ROOT_DIR)
