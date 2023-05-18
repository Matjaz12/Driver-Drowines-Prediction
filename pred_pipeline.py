import argparse
import os
import re

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from retinaface.pre_trained_models import get_model

from model.load_data import get_test_transform, label_ds
from model.resmasknet import load_resmasknet
from model.resnet50 import load_resnet50


class DriverStatePredictor:
    """
    Driver State Predictor:
    1. Detects drivers face
    2. Predict driver state (alert, microsleep, yawning)
    """
    def __init__(
        self, face_detector, driver_state_model, transform=get_test_transform()
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = face_detector
        self.driver_state_model = driver_state_model.to(self.device)
        self.transform = transform

    def __call__(self, img):
        """Prediction pipeline."""
        bbox, img_face = self._detect_face(img)
        if bbox is None: return None, None, None
        driver_state, confidence = self._predict_driver_state(img_face)
        return bbox, driver_state, confidence

    def _detect_face(self, img):
        """Detect driver face on current frame."""
        # detect faces on current frame
        preds = self.face_detector.predict_jsons(img)
        preds = ([p for p in preds if p["score"] != -1])
        if len(preds) == 0: return None, None

        # detect the driver's face, i.e the face bbox with max. area        
        bbox_area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        idx = np.argmax([bbox_area(pred["bbox"]) for pred in preds])
        bbox = preds[idx]["bbox"]
        left, top, right, bottom = bbox
        img_face = img[int(top):int(bottom), int(left):int(right)]
        return bbox, img_face
    
    def _predict_driver_state(self, img_face):
        """Predict driver state given a cropped image of the face."""
        img_face = self.transform(image=img_face)["image"]
        img_face = ToTensorV2()(image=img_face)["image"].unsqueeze(dim=0)
        img_face = img_face.to(self.device)
        preds = self.driver_state_model(img_face)
        pred = preds.argmax(dim=1).cpu().item()
        confidence = torch.softmax(preds, dim=1)[0][pred].cpu().item()
        return pred, confidence


def draw_prediction(frame, dsp):
    """Display prediction on frame."""
    bbox, driver_state, confidence = dsp(frame) # predict driver state
    if bbox is None: return frame, None, None
    topleft = (int(bbox[0]), int(bbox[1]))
    bottomright = (int(bbox[2]), int(bbox[3]))
    frame = cv2.rectangle(frame, topleft, bottomright, (255, 0, 0), 2)
    frame = cv2.putText(
        frame, f"{label_ds[driver_state]}: {round(confidence, 2)}", (topleft[0], topleft[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame, bbox, driver_state
    

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="resnet50")
    args = parser.parse_args()
    
    # load models
    print(f"loading model {args.model}...", flush=True)
    if args.model == "resnet50":
        model = load_resnet50(n_classes=3)
    elif args.model == "resmasknet":
        model = load_resmasknet(n_classes=3)
    else:
        raise Exception(f"Model: {args.model} not supported!")
    
    state_dict_path =f"./model/{args.model}/{args.model}_ds.pt"
    print(f"loading state dictionary from {state_dict_path}", flush=True)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    retinaface = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    retinaface.eval()

    # define the pred. pipeline object.
    dsp = DriverStatePredictor(face_detector=retinaface, driver_state_model=model)
    img_path = os.path.join(f"./dataset-cover.jpg")
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, bbox, driver_state = draw_prediction(frame, dsp)
    cv2.imwrite(f"./dataset-cover-pred.jpg",
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    exit()

    # open web-cam and make predictions
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        
        frame, bbox, driver_state = draw_prediction(frame, dsp)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
