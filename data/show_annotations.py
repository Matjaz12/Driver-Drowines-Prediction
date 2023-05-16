import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_bboxes(img, bboxes):
    """Draw bounding boxes on image."""
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox][:4]
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return img


def draw_landmarks(img, landmarks, radius=3):
    """Draw landmarks on image."""
    for landmark in landmarks:
        cv2.circle(img, (int(landmark[0]), int(landmark[1])), radius, (0, 255, 0), -1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to video frames")
    parser.add_argument("annotations_path", help="path to annotations file")
    args = parser.parse_args()

    with open(args.annotations_path) as file:
        annotations = json.load(file)

    # iterate over frames in data_path
    for frame in tqdm(os.listdir(args.data_path)):
        if ".jpg" not in frame:
            continue
        img = cv2.imread(os.path.join(args.data_path, frame))
        if frame not in annotations:
            print(f"frame not in annotations, skipping frame: {frame}")
            continue
        bbox = annotations[frame]["bbox"] if "bbox" in annotations[frame] else None
        plt.figure(figsize=(10, 10))
        plt.title(
            f"annotation for frame: {frame}, driver_state: {annotations[frame]['driver_state'] if 'driver_state' in annotations[frame] else None}")
        img = draw_bboxes(img, [bbox]) if bbox else img
        img = draw_landmarks(img, annotations[frame]["landmarks"]) if "landmarks" in annotations[frame] else img
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.show()
