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

model = "faster"
# Dataset guardado en pkl file. El script para generar el pickle está en GitHub
pkl_file_train = open('./train_KITTI-MOTS_dataset_server.pkl', 'rb')
pkl_file_val = open('./validation_KITTI-MOTS_dataset_server.pkl', 'rb')

dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
dataset_dicts_val = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
# random.shuffle(dataset_dicts)

# val_size = round(len(dataset_dicts)*0.2)
# train_size = len(dataset_dicts)-val_size

# dataset_dicts_train = dataset_dicts[0:train_size]
# dataset_dicts_val = dataset_dicts[train_size:]

def kitti_mots_dataset(d):
    if d == "kitti-mots-dataset-train":
        return dataset_dicts_train
    else:
        return dataset_dicts_val

# TRAIN_PATH_IMAGES = '../mcv/datasets/KITTI-MOTS/training/image_02/'
# all_images = []
# folders = os.listdir(TRAIN_PATH_IMAGES)

for d in ["train", "val"]:
  DatasetCatalog.register("kitti-mots-dataset-" + d, lambda d=d: kitti_mots_dataset("kitti_mots_dataset_" + d))
  MetadataCatalog.get("kitti-mots-dataset-" + d).set(thing_classes=["none", "Car" ,"Pedestrian"])


kitti_mots_metadata_train = MetadataCatalog.get("kitti-mots-dataset-train")
kitti_mots_metadata_val = MetadataCatalog.get("kitti-mots-dataset-val")


cfg = get_cfg()
if model == "retinanet":
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
else:
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kitti-mots-dataset-train",)
cfg.DATASETS.TEST = ("kitti-mots-dataset-train",)
cfg.OUTPUT_DIR = './output_kittimots_all'
cfg.DATALOADER.NUM_WORKERS = 1
if model == "retinanet":
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml") 
else:
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (1000, 3000)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
# cfg.MODEL.WEIGHTS = './output_kittimots_all/model_final.pth'

# predictor = DefaultPredictor(cfg)


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


evaluator = COCOEvaluator("kitti-mots-dataset-val", cfg, False, output_dir='./output_kittimots_all')
trainer.test(cfg, trainer.model, evaluators=[evaluator])