import os
import argparse
import json
import re
import cv2
import torch
from retinaface.pre_trained_models import get_model
from pred_pipeline import DriverStatePredictor, draw_prediction
from model.resmasknet import load_resmasknet
from model.resnet50 import load_resnet50
# from model.ensemble import load_ensemble
from torchvision.ops import box_iou
from torchmetrics.classification import (Accuracy, ConfusionMatrix, F1Score,
                                         Precision, Recall)


def pipeline_test(dsp, data_path):
    """Test the entire pipeline on out-of-distribution (ood) data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs_path = os.path.join(data_path, "imgs")
    annotations = json.load(open(os.path.join(imgs_path, "_annotations.coco.json")))
    n_detector_fails = 0 

    preds, labels = {"driver_state": [], "bbox": []}, {"driver_state": [], "bbox": []}
    for idx, img in enumerate(annotations["images"]):
        # Read frame and label
        img_path = os.path.join(imgs_path, img["file_name"])
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label = annotations["annotations"][idx]
        if label["id"] != img["id"]:
            raise Exception("Sample miss-match!")

        # Make predictions and save labeled frame
        frame, bbox, driver_state = draw_prediction(frame, dsp)
        if bbox is None:
            n_detector_fails += 1
            print("face-detector failed!")
            continue
        cv2.imwrite(os.path.join(data_path, "preds", img["file_name"]),
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preds["driver_state"].append(int(driver_state))
        preds["bbox"].append(bbox)

        # Save gt. label
        labels["driver_state"].append(label["category_id"] - 1)
        bbox = label["bbox"]  # [top left x position, top left y position, width, height]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        labels["bbox"].append(bbox)

    # compute mean intersection over union
    pred_bboxes = torch.tensor(preds["bbox"]).view(len(preds["bbox"]), 4)
    gt_bboxes = torch.tensor(labels["bbox"]).view(len(labels["bbox"]), 4)
    ious = torch.diagonal(box_iou(pred_bboxes, gt_bboxes))
    mean_iou = torch.mean(ious).item()
    print(f"(face-detector) mean_iou: {mean_iou}, n_detector_fails: {n_detector_fails}")
          
    # compute the acc, macro f1-score, precision and recall
    acc_fun = Accuracy(task="multiclass", num_classes=3).to(device)
    f1_fun = F1Score(task="multiclass", average="macro", num_classes=3).to(device)
    precision_fun = Precision(task="multiclass", average='macro', num_classes=3).to(device)
    recall_fun = Recall(task="multiclass", average='macro', num_classes=3).to(device)
    preds_all = torch.tensor(preds["driver_state"]).to(device)
    labels_all = torch.tensor(labels["driver_state"]).to(device)
    acc = acc_fun(preds_all, labels_all)
    f1 = f1_fun(preds_all, labels_all)
    precision = precision_fun(preds_all, labels_all)
    recall = recall_fun(preds_all, labels_all)
    print(f"(driver-state predictor) acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")
    

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="resnet50")
    parser.add_argument("--data_path", type=str, required=True, default="resnet50")
    args = parser.parse_args()
    
    # load models
    print(f"loading model {args.model}...", flush=True)
    if args.model == "resnet50":
        model = load_resnet50(n_classes=3)
    elif args.model == "resmasknet":
        model = load_resmasknet(n_classes=3)
    # elif args.model == "ensemble":
    #     model = load_ensemble()
    else:
        raise Exception(f"Model: {args.model} not supported!")
    
    if args.model != "ensemble":
        state_dict_path =f"./model/{args.model}/{args.model}_ds.pt"
        print(f"loading state dictionary from {state_dict_path}", flush=True)
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()
    
    retinaface = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    retinaface.eval()

    # define the pred. pipeline object.
    dsp = DriverStatePredictor(face_detector=retinaface, driver_state_model=model)
    print(f"Making predictions for images in: {args.data_path}", flush=True)
    pipeline_test(dsp, args.data_path)
    