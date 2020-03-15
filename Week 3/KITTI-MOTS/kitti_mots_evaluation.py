import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import glob
import cv2
import os
import random
import pickle
from tqdm import tqdm
from google.colab.patches import cv2_imshow

import torch, torchvision

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from detectron2.data import DatasetCatalog, MetadataCatalog

def main():
    pkl_file_train = open('../KITTI-MOTS/train_KITTI-MOTS_dataset_local.pkl', 'rb')
    pkl_file_val = open('../KITTI-MOTS/validation_KITTI-MOTS_dataset_local.pkl', 'rb')

    dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
    dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')

    def kitti_mots_dataset(d):
        if d == "kitti-mots-dataset-train":
            return dataset_dicts_train
        else:
            return dataset_dicts_validation

    for d in ["kitti-mots-dataset-train", "kitti-mots-dataset-validation"]:
        DatasetCatalog.register(d, lambda d=d: kitti_mots_dataset(d))
        MetadataCatalog.get(d).set(thing_classes=["none", "Car", "Pedestrian"])

    kitti_mots_metadata_train = MetadataCatalog.get("kitti-mots-dataset_train")
    kitti_mots_metadata_validation = MetadataCatalog.get("kitti-mots-dataset-validation")

    retinanet = False

    # Load MODEL and Configuration
    PATH = './output'
    cfg = get_cfg()

    if retinanet:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("kitti-mots-dataset-train",)
    cfg.DATASETS.TEST = ("kitti-mots-dataset-validation",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.OUTPUT_DIR = PATH

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
        
    # Evaluation
    evaluator = COCOEvaluator("kitti-mots-dataset-validation", cfg, False, output_dir=PATH)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

if __name__ == "__main__":
    main()
