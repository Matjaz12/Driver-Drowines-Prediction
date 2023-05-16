# FL-DDD (Frame Level Driver Drowsiness Detection) 

Final (annotated) dataset is available here: [Frame Level Driver Drowsiness Prediction](https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection)

## Dataset Annotation

We used the [NITYMED](https://datasets.esdalab.ece.uop.gr/download-files/) dataset. NITYMED contains videos during which drivers either yawn of have a microsleep. We annotate each frame with a bounding box (around the drivers face), a set of facial keypoints and a label (`[alert, microsleep, yawning]`) denoting the drivers state at the current frame. To detect face bounding boxes and landmarks we used the [retinaface model](https://github.com/serengil/retinaface). Detected bounding boxes are checked by a human. Human also provides a `drivers_state` label to each frame.

## Out Of Distribution (OOD) Dataset

To evaluate models ability to generalize to out of distribution data, we used the following dataset:[roboflow-driver-drowsiness-detection](https://universe.roboflow.com/augmented-startups/drowsiness-detection-cntmz/dataset/1).