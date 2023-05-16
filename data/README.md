# FL-DDPD (Frame Level Driver Drowsiness Prediction Dataset) 

Final (annotated) dataset is available here: [Frame Level Driver Drowsiness Prediction](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection)

## Dataset Annotation

We used the  [NITYMED](https://datasets.esdalab.ece.uop.gr/download-files/) dataset. NITYMED contains videos during which drivers either yawn of have a microsleep. We annotate each frame with a bounding box (around the drivers face), a set of facial keypoints and a label (`[alert, microsleep, yawning]`) denoting the drivers state at the current frame. To detect face bounding boxes and landmarks we used the [retinaface model](https://github.com/serengil/retinaface). Detected bounding boxes are checked by a human. Human also provides a `drivers_state` label to each frame.