# Frame Level Classification Dataset (FLC Dataset)

## Dataset Annotation

We used the  [NITYMED](https://datasets.esdalab.ece.uop.gr/download-files/) dataset. NITYMED contains videos during which drivers either yawn of have a microsleep. We annotate each frame with a bounding box (around the drivers face), a set of facial keypoints and a label (`[alert, microsleep, yawning]`) denoting the drivers state at the current frame. To detect face bounding boxes and landmarks we used the [retinaface model](https://github.com/serengil/retinaface). Detected bounding boxes are checked by a human. Human also provides a `drivers_state` label to each frame.

## Train, Validation and Test split

In order to insure that the validation and test set is diverse enough and not too similar to the trianing set, we sample full intervals of the driver states and use the entire interval in the validation and test set.