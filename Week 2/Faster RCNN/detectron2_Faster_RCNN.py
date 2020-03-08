import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# get image
# TRAIN_PATH = 'MIT_split/train/'
# TEST_PATH = 'MIT_split/test/'
TRAIN_PATH = '../mcv/datasets/MIT_split/train/'
TEST_PATH = '../mcv/datasets/MIT_split/test/'

# Create config
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# # cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 # set threshold for this model
# cfg.MODEL.RETINANET.NUM_CLASSES = 500

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 # set threshold for this model
# cfg.MODEL.RETINANET.NUM_CLASSES = 500

# Create predictor
predictor = DefaultPredictor(cfg)

# Read images
folders = []
nimg = 0

for item in glob.glob(TEST_PATH + '/*'):
    folders.append(item.split('/')[-1])
    
# for folder in folders:
images = glob.glob(TEST_PATH + "street" + '/*') 
for image in images: 
    nimg = nimg + 1
    print("Image: ", nimg)
    
    im = cv2.imread(image)
    im = cv2.resize(im,(1024,1024))
    # Make prediction
    outputs = predictor(im)
    
    # Visualize result
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    cv2.imwrite("object_detection/RetinaNET/img" + str(nimg) + ".jpg", v.get_image()[:, :, ::-1])