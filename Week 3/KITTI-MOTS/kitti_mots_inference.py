import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

TRAIN_PATH = '../KITTI-MOTS/training/image_02/'
images_1 = glob.glob(TRAIN_PATH + '0010/*.png')
images_2 = glob.glob(TRAIN_PATH + '0013/*.png')
images_3 = glob.glob(TRAIN_PATH + '0005/*.png')

random_images = []

random_images.append(random.sample(images_1, 1)[0])
random_images.append(random.sample(images_2, 1)[0])
random_images.append(random.sample(images_3, 1)[0])

retinanet = False

# Create config
cfg = get_cfg()

if retinanet:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
else:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

thresholds = [0.5, 0.7, 0.85]
if retinanet:
    net = 'retinanet'
else:
    net = 'faster_rcnn'
        
# for folder in folders:
for image in random_images: 
    for threshold in thresholds:
        if retinanet:
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        else:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # Create predictor
        predictor = DefaultPredictor(cfg)
        print(image)
        im = cv2.imread(image)

        # Make prediction
        outputs = predictor(im)
            
        # Visualize result
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
        cv2.imshow('Image', v.get_image()[:, :, ::-1])
        cv2.imwrite('./Week3/task_b/thres_' + str(threshold) + '_' + net + '_' + image.split('/')[-1], v.get_image()[:, :, ::-1])