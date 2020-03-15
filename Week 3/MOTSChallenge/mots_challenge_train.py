import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import glob
import cv2
import os
import random
# from google.colab.patches import cv2_imshow

import torch, torchvision

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import pickle

pkl_file_train = open('train_MOTSChallenge_dataset_server.pkl', 'rb')
pkl_file_val = open('validation_MOTSChallenge_dataset_server.pkl', 'rb')

dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
dataset_dicts_val = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')

def mots_challenge_dataset(path):
    if path == "mots_challenge_dataset_train":
      return dataset_dicts_train
    else:
      return dataset_dicts_val

TRAIN_PATH = '../mcv/datasets/MOTSChallenge/train/images/'

for d in ["train", "val"]:
  DatasetCatalog.register("mots_challenge_dataset_" + d, lambda d=d: mots_challenge_dataset("mots_challenge_dataset_" + d))
  MetadataCatalog.get("mots_challenge_dataset_" + d).set(thing_classes=["Pedestrian", "None", "None"])

mots_challenge_metadata_train = MetadataCatalog.get("mots_challenge_dataset_train")
mots_challenge_metadata_validation = MetadataCatalog.get("mots_challenge_dataset_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mots_challenge_dataset_train",)
cfg.DATASETS.TEST = ("mots_challenge_dataset_val", )
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") 
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml") 
cfg.OUTPUT_DIR = ("./output/faster_rcnn")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (1000, 3000)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6 # set threshold for this model

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("mots_challenge_dataset_val", cfg, False, output_dir="./output/faster_rcnn/")
trainer.test(cfg, trainer.model, evaluators=[evaluator])