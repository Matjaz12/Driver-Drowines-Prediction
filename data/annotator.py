"""
python3 annotator.py frames/P1043080_720 frames/P1043080_720/annotations.json
"""

import cv2
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import json
from show_annotations import draw_bboxes, draw_landmarks


def display_gui(img, bbox, landmarks, frame, seq):
    """Display the image and buttons for data annotating"""

    # display the image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title(f"({seq}) annotation for frame: {frame}")
    img = draw_bboxes(img, [bbox])
    img = draw_landmarks(img, landmarks, radius=1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])

    # Add the buttons to the figure
    button_width = 0.165
    button_height = 0.05
    button_padding = 0
    buttons, buttons_text = [], ["alert", "microsleep", "yawning", "remove", "back"]
    for i in range(len(buttons_text)):
        button = plt.axes(
            [0.1 + i * (button_width + button_padding), 0.1, button_width, button_height])
        button_obj = plt.Button(button, buttons_text[i])
        button_obj.label.set_text(f"{buttons_text[i]} ({i + 1})")
        buttons.append(button_obj)

    # Add accelerator keys to the buttons
    fig.canvas.mpl_connect(
        'key_press_event', lambda event: button_on_key_press(event, buttons))

    def button_on_key_press(event, buttons):
        for button in buttons:
            if button.label.get_text().endswith('({})'.format(event.key)):
                global label_global
                label_global = button.label.get_text().split(" ")[0]
                plt.close(fig)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to video frames")
    parser.add_argument("annotations_path", help="path to annotations file")

    args = parser.parse_args()

    with open(args.annotations_path) as file:
        annotations = json.load(file)

    # capture frames
    frames = []
    for frame in os.listdir(args.data_path):
        if ".jpg" in frame:
            frames.append(frame)
    frames.sort(key=lambda x: int(x.split(".")[0].split("frame")[1]))

    # iterate over frames and display gui
    annotations_new = {}
    label_global = None
    idx = 0

    while idx < len(frames):
        frame = frames[idx]
        img = cv2.imread(os.path.join(args.data_path, frame))

        if frame not in annotations:
            print(f"frame not in annotations, skipping frame: {frame}")
            idx += 1
            continue

        bbox = annotations[frame]["bbox"]
        landmarks = annotations[frame]["landmarks"]
        display_gui(img, bbox, landmarks, frame, args.data_path)

        if label_global == "remove":
            print(f"removing frame and annotation: {frame}")
            if frame in annotations_new:
                print(f"deleting prev. added {frame} annot.")
                del annotations_new[frame]
            idx += 1
            continue

        if label_global == "back":
            idx -= 1 if idx > 0 else 0
            continue

        # update annotations
        annotations_new[frame] = annotations[frame]  # copy the old annotations
        annotations_new[frame]["driver_state"] = label_global

        # save the new annotations
        with open(f'{"/".join(args.annotations_path.split("/")[:-1])}/annotations_final.json', "w") as file:
            json.dump(annotations_new, file)

        idx += 1
