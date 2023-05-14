"""
python3 annotator.py frames/P1043080_720 frames/P1043080_720/annotations.json
"""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
from show_annotations import draw_bboxes, draw_landmarks
from tqdm import tqdm


def display_gui(img, bbox, landmarks, driver_state, frame, seq):
    """Display the image and buttons for data annotating"""

    # display the image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title(f"({seq}) annotation for frame: {frame}, driver_state: {driver_state}")
    img = draw_bboxes(img, [bbox]) if bbox else img
    img = draw_landmarks(img, landmarks, radius=1) if landmarks else img
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


label_global = None
def annotate(args):
    """Annotate frames."""

    if os.path.exists(f"{'/'.join(args.annotations_path.split('/')[:-1])}/annotations_new.json") and args.start_idx != None:
        print("annotations_new.json already exists, opening that instead.")
        annot_path = f"{'/'.join(args.annotations_path.split('/')[:-1])}/annotations_new.json"
    else:
        annot_path = args.annotations_path
    with open(annot_path) as file:
        annotations = json.load(file)

    # capture frames
    frames = []
    for frame in os.listdir(args.data_path):
        if ".jpg" in frame:
            frames.append(frame)
    frames.sort(key=lambda x: int(x.split(".")[0].split("frame")[1]))

    # iterate over frames and display gui
    annotations_new = {} if args.start_idx == None else annotations
    idx = 0 if args.start_idx == None else args.start_idx

    while idx < len(frames):
        frame = frames[idx]
        img = cv2.imread(os.path.join(args.data_path, frame))

        if frame not in annotations:
            print(f"frame not in annotations, skipping frame: {frame}")
            idx += 1
            continue

        bbox = annotations[frame]["bbox"] if "bbox" in annotations[frame] else None
        landmarks = annotations[frame]["landmarks"] if "landmarks" in annotations[frame] else None
        driver_state = annotations[frame]["driver_state"] if "driver_state" in annotations[frame] else None
        display_gui(img, bbox, landmarks, driver_state, frame, args.data_path)

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
        print(f"{frame}, label: {label_global}")
        annotations_new[frame] = annotations[frame]  # copy the old annotations
        annotations_new[frame]["driver_state"] = label_global

        # save the new annotations
        with open(f'{"/".join(args.annotations_path.split("/")[:-1])}/annotations_new.json', "w") as file:
            json.dump(annotations_new, file)

        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to video frames")
    parser.add_argument("annotations_path", help="path to annotations file")
    parser.add_argument("--start_idx", help="start index for annotating", type=int, required=False, default=None)
    args = parser.parse_args()
    annotate(args)
