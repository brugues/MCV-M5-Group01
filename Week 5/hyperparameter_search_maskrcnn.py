import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob
import pickle
import json
import os
import matplotlib.pyplot as plt
from KITTITrainer import *

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_loss_curve(cfg, model_name):
    experiment_metrics = load_json_arr(os.path.join(cfg.OUTPUT_DIR, 'metrics.json'))

    training_losses = []
    validation_losses = []
    idx = []

    for line in experiment_metrics:
        print(line)
        if 'total_loss' in line.keys() and 'validation_loss' in line.keys():
            idx.append(line['iteration'])
            training_losses.append(line['total_loss'])
            validation_losses.append(line['validation_loss'])

    print(idx)
    print(validation_losses)
    print(training_losses)

    plt.plot(idx, validation_losses, label="Validation Loss")
    plt.plot(idx, training_losses, label="Training Loss")
    plt.title('Loss curves for model ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Validation_Loss')
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'loss_curve.png'))

# MODEL PARAMETERS TO ADJUST
MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL_NAME = MODEL.split('.')[0]
USE_KITTI_MOTS_WEIGHTS = False
TRAIN = True
TRAIN_SET = 'MOTSChallenge'
VAL_SET = 'KITTI-MOTS'
GET_VALIDATION_PLOTS = False
DATASETS_REGISTERED = False

# TRAINING PARAMETERS TO ADJUST
NUMBER_OF_EPOCHS = 3500
SCORE_THRESHOLD = 0.6
BATCH_SIZE = 256
MIN_STEP = 1000
MAX_STEP = NUMBER_OF_EPOCHS
EVAL_PERIOD = 50

""" LR Scheduler Options"""
LEARNING_RATE_SCHEDULER = 'WarmupMultiStepLR'  # 'WarmupCosineLR' 'WarmupMultiStepLR'. WarmupMultiStepLR is the DEFAULT
BASE_LEARNING_RATE = 0.00075
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0001

""" Input size of the image """
INPUT_SIZE_ENABLE = False
MIN_SIZE_TRAIN = 800
MAX_SIZE_TRAIN = 2500
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 2500

""" Image crop for data augmentation """
CROP_ENABLE = False
CROP_TYPE = 'relative_range'  # 'relative', 'relative_range', 'absolute' relative_range is the DEFAULT
CROP_SIZE = [0.9, 0.9]

PATH_MOTSCHALLENGE = './output/MOTSChallenge/' + MODEL_NAME
PATH_KITTI_MOTS = './output/KITTI_MOTS/' + MODEL_NAME
KITTI_MOTS_WEIGHTS = PATH_KITTI_MOTS + '/model_final.pth'

TRAIN_PKL = './train_' + TRAIN_SET + '_dataset_server.pkl'
VAL_PKL = './validation_' + VAL_SET + '_dataset_server.pkl'
pkl_file_train = open(TRAIN_PKL, 'rb')
dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
pkl_file_val = open(VAL_PKL, 'rb')
dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')


def dataset(d):
    if d == "dataset-train":
        return dataset_dicts_train
    else:
        return dataset_dicts_validation


def main():
    print('Starting ')

    if TRAIN:
        print('Training with the following config \n')
        print('Base Learning Rate: ', BASE_LEARNING_RATE)
        print('Momentum: ', MOMENTUM)
        print('Weight Decay', WEIGHT_DECAY)
        print('Number of Epochs: ', NUMBER_OF_EPOCHS)
        print('Crop: ', CROP_ENABLE)
        if CROP_ENABLE:
            print('Crop size and type: ', CROP_SIZE, ' ', CROP_TYPE)
        if INPUT_SIZE_ENABLE:
            print('Input size min: ', MIN_SIZE_TEST)
            print('Input size max: ', MAX_SIZE_TEST)
    else:
        print('Inference on ', VAL_SET)

    print('\n\n')

    # Configuration
    cfg = get_cfg()

    if GET_VALIDATION_PLOTS:
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

    if VAL_SET == 'KITTI-MOTS':
        cfg.OUTPUT_DIR = PATH_KITTI_MOTS
    elif VAL_SET == 'MOTSChallenge':
        cfg.OUTPUT_DIR = PATH_MOTSCHALLENGE
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))

    cfg.DATALOADER.NUM_WORKERS = 2
    if USE_KITTI_MOTS_WEIGHTS:
        cfg.MODEL.WEIGHTS = KITTI_MOTS_WEIGHTS
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.LR_SCHEDULER_NAME = LEARNING_RATE_SCHEDULER
    cfg.SOLVER.BASE_LR = BASE_LEARNING_RATE
    cfg.SOLVER.MAX_ITER = NUMBER_OF_EPOCHS
    #cfg.SOLVER.MOMENTUM = MOMENTUM
    #cfg.SOLVER.WEIGHT_DECAY = WEIGHT_DECAY
    #cfg.SOLVER.STEPS = (MIN_STEP, MAX_STEP)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    if TRAIN_SET == 'KITTI-MOTS':
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    if INPUT_SIZE_ENABLE:
        cfg.INPUT.MIN_SIZE_TRAIN = MIN_SIZE_TRAIN
        cfg.INPUT.MAX_SIZE_TRAIN = MAX_SIZE_TRAIN
        cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
        cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    if CROP_ENABLE:
        cfg.INPUT.CROP.ENABLE = CROP_ENABLE
        cfg.INPUT.CROP.TYPE = CROP_TYPE
        cfg.INPUT.CROP.SIZE = CROP_SIZE

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Finish configuration depending on the procedure we are performing
    datasets = ['dataset-train', 'dataset-val']
    cfg.DATASETS.TRAIN = (datasets[0],)
    cfg.DATASETS.TEST = (datasets[1],)

    # Register the datasets
    registered = DATASETS_REGISTERED

    if not registered:
        for d in datasets:
            DatasetCatalog.register(d, lambda d=d: dataset(d))
            MetadataCatalog.get(d).set(thing_classes=["Pedestrian", "None", "Car"])

    kitti_mots_metadata_train = MetadataCatalog.get(datasets[0])
    kitti_mots_metadata_validation = MetadataCatalog.get(datasets[1])

    # Set-up trainer
    if GET_VALIDATION_PLOTS:
        trainer = KITTITrainer(cfg)
    else:
        trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)

    # Train if wanted
    if TRAIN:
        print('Start training')
        trainer.train()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    # Start evaluation
    if GET_VALIDATION_PLOTS:
        trainer.test(cfg, trainer.model)
        plot_loss_curve(cfg, MODEL_NAME)
    else:
        evaluator = COCOEvaluator(datasets[1], cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, datasets[1])
        inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":
    main()