# Frame Level Driver Drowsiness Prediction

Implementation of a driver drowsiness prediction system. The [Prediction Pipeline](./pred_pipeline.py) has two stages. In stage one we detect the drivers face, using a pre-trained [retinaface](https://github.com/serengil/retinaface) model, the detected face is than fed into a [ResMaskNet](https://py-feat.org/pages/models.html) model, which was pretrained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), and finetuned for Driver Drowsiness Prediction. We fine-tune the model on a custom dataset called [FL-DDPD](./data/README.md).