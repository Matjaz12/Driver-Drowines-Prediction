import cv2
import matplotlib.pyplot as plt
import os
import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_file", help="annotations file")
    args = parser.parse_args()

    annots = json.load(open(args.annotations_file))

    for frame, label in annots.items():
        img = cv2.imread(frame)
        plt.title(f"frame: {frame}, driver_state: {label['driver_state']}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()