import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob
import pickle
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# MODEL PARAMETERS TO ADJUST
MODEL_NAME = 'Cityscapes/mask_rcnn_R_50_FPN'
USE_KITTI_MOTS_WEIGHTS = False
TRAIN = True
TRAIN_SET = 'MOTSChallenge'
VAL_SET = 'KITTI-MOTS'

# TRAINING PARAMETERS TO ADJUST
NUMBER_OF_EPOCHS = 3500
SCORE_THRESHOLD = 0.6
BATCH_SIZE = 256
MIN_STEP = 1500
MAX_STEP = NUMBER_OF_EPOCHS

""" LR Scheduler Options"""
LEARNING_RATE_SCHEDULER = 'WarmupMultiStepLR' # 'WarmupCosineLR'
BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001

""" Input size of the image """
MIN_SIZE_TRAIN = 800
MAX_SIZE_TRAIN = 1333
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 1333

""" Image crop for data augmentation """
CROP_ENABLE = False
CROP_TYPE = 'relative_range' # 'relative', 'relative_range', 'absolute' 
CROP_SIZE = [0.9, 0.9]


PATH_MOTSCHALLENGE = './results_MOTSChallenge/' + MODEL_NAME
PATH_KITTI_MOTS = './results_KITTI_MOTS/' + MODEL_NAME
KITTI_MOTS_WEIGHTS = PATH_KITTI_MOTS + '/model_final.pth'


if TRAIN:
    TRAIN_PKL = './train_' + TRAIN_SET + '_dataset_server.pkl'
    VAL_PKL = './validation_' + VAL_SET + '_dataset_server.pkl'
    pkl_file_train = open(TRAIN_PKL, 'rb')
    dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
    pkl_file_val = open(VAL_PKL, 'rb')
    dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
else:
    VAL_PKL = './validation_' + VAL_SET + '_dataset_server.pkl'
    pkl_file_val = open(VAL_PKL, 'rb')
    dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')


def dataset(d):
    if d == "dataset-train":
        return dataset_dicts_train
    else:
        return dataset_dicts_validation


def main():
    cfg = get_cfg()
    if VAL_SET == 'KITTI-MOTS':
        cfg.OUTPUT_DIR = PATH_KITTI_MOTS
    elif VAL_SET == 'MOTSChallenge':
        cfg.OUTPUT_DIR = PATH_MOTSCHALLENGE
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_NAME + ".yaml"))

    if not TRAIN:
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        thing_classes = metadata.thing_classes
        del metadata 

    cfg.DATALOADER.NUM_WORKERS = 2
    if USE_KITTI_MOTS_WEIGHTS:
        cfg.MODEL.WEIGHTS = KITTI_MOTS_WEIGHTS
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_NAME + ".yaml") 
    cfg.SOLVER.IMS_PER_BATCH = 2
    
    # Cfg params
    cfg.SOLVER.LR_SCHEDULER_NAME = LEARNING_RATE_SCHEDULER
    cfg.SOLVER.BASE_LR = BASE_LEARNING_RATE
    cfg.SOLVER.MAX_ITER = NUMBER_OF_EPOCHS   
    cfg.SOLVER.MOMENTUM = MOMENTUM
    cfg.SOLVER.WEIGHT_DECAY = WEIGHT_DECAY
    cfg.SOLVER.STEPS = (MIN_STEP, MAX_STEP)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE   
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD  
    
    cfg.INPUT.MIN_SIZE_TRAIN = MIN_SIZE_TRAIN
    cfg.INPUT.MAX_SIZE_TRAIN = MAX_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST
    
    # cfg.INPUT.CROP = {"ENABLED": CROP_ENABLE}
    # cfg.INPUT.CROP.TYPE = CROP_TYPE
    # cfg.INPUT.CROP.SIZE = CROP_SIZE
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Finish configuration depending on the procedure we are performing
    if TRAIN:
        datasets = ['dataset-train', 'dataset-validation']
        cfg.DATASETS.TRAIN = ('dataset-train',)
        cfg.DATASETS.TEST = ("dataset-validation", )
    else:
        datasets = ['dataset-validation']
        cfg.DATASETS.TRAIN = ('dataset-validation',)

    # Register the datasets
    for d in datasets:
        DatasetCatalog.register(d, lambda d=d: dataset(d))
        if TRAIN:
            MetadataCatalog.get(d).set(thing_classes=["Car", "Pedestrian"])
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        else:
            MetadataCatalog.get(d).set(thing_classes=thing_classes)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

    # Set-up trainer
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    # Train if wanted
    if TRAIN:
        print('Start training')
        trainer.train()

    # Start evaluation
    evaluator = COCOEvaluator("dataset-validation", cfg, False, output_dir=cfg.OUTPUT_DIR)
        
    if TRAIN:
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
    else:
        val_loader = build_detection_test_loader(cfg, "dataset-validation")
        inference_on_dataset(trainer.model, val_loader, evaluator)
    
if __name__ == "__main__":
    main()