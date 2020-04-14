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


# PARAMETERS
TRAIN = True
REGISTERED = False
MODEL = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
MODEL_NAME = MODEL.split('.')[0]
THRESHOLD = 0.6

# TRAINING PARAMETERS TO ADJUST
# NUMBER_OF_EPOCHS = 5000
SCORE_THRESHOLD = 0.6
BATCH_SIZE = 256
MIN_STEP = 2000
# MAX_STEP = NUMBER_OF_EPOCHS

""" LR Scheduler Options"""
LEARNING_RATE_SCHEDULER = 'WarmupMultiStepLR'  # 'WarmupCosineLR' 'WarmupMultiStepLR'. WarmupMultiStepLR is the DEFAULT
# BASE_LEARNING_RATE = 0.00075
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0001

""" Input size of the image """
MIN_SIZE_TRAIN = 800
MAX_SIZE_TRAIN = 2500
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 2500

pkl_file_train = open('./pickles/train_real_KITTI-MOTS_dataset_server.pkl', 'rb')
dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
pkl_file_train_virtual = open('./pickles/train_synthetic_vKITTI_dataset_server.pkl', 'rb')
dataset_dicts_train_virtual = pickle.load(pkl_file_train_virtual, fix_imports=True, encoding='ASCII', errors='strict')
pkl_file_val = open('./pickles/validation_real_KITTI-MOTS_dataset_server.pkl', 'rb')
dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
pkl_file_test = open('./pickles/test_real_KITTI-MOTS_dataset_server.pkl', 'rb')
dataset_dicts_test = pickle.load(pkl_file_test, fix_imports=True, encoding='ASCII', errors='strict')

def kitti_mots_dataset(d):
    if d == "kitti-mots-dataset-train":
        global dataset_dicts_train_virtual, dataset_dicts_train 
        N = int(len(dataset_dicts_train) * (10 / 100))
        # random.shuffle(dataset_dicts_train)
        dataset_dicts_train_virtual += dataset_dicts_train[0:N]
        return dataset_dicts_train_virtual
    else:
        return dataset_dicts_validation


def main(BASE_LEARNING_RATE, NUMBER_OF_EPOCHS):
    print("Starting! \n")
    print("Model: ", MODEL, "\n")
    print("LR: ", BASE_LEARNING_RATE, "\n")
    print("MAX_ITER: ", NUMBER_OF_EPOCHS, "\n")
    print("THRESHOLD: ", THRESHOLD, "\n")

    if TRAIN:
        PATH = './results_taskb_sergio/' + MODEL_NAME + '/LR_' + str(BASE_LEARNING_RATE) + '_MAXITER_' + str(NUMBER_OF_EPOCHS) + '_REAL_10_TEST'
    else:
        PATH = './results_taskb_sergio/' + MODEL_NAME + '/TESTING'

    cfg = get_cfg()
    cfg.OUTPUT_DIR = PATH
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    modelzoo_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = BASE_LEARNING_RATE
    cfg.SOLVER.MAX_ITER = NUMBER_OF_EPOCHS
    
    cfg.SOLVER.LR_SCHEDULER_NAME = LEARNING_RATE_SCHEDULER
    cfg.SOLVER.MOMENTUM = MOMENTUM
    cfg.SOLVER.WEIGHT_DECAY = WEIGHT_DECAY
    cfg.SOLVER.STEPS = (MIN_STEP, NUMBER_OF_EPOCHS)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    cfg.INPUT.MIN_SIZE_TRAIN = MIN_SIZE_TRAIN
    cfg.INPUT.MAX_SIZE_TRAIN = MAX_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    # if TRAIN:
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 101

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATASETS.TRAIN = ('kitti-mots-dataset-train',)
    cfg.DATASETS.TEST = ("kitti-mots-dataset-validation", )

    # Register the datasets
    if not REGISTERED:
        # Register the datasets
        for d in ['kitti-mots-dataset-train', 'kitti-mots-dataset-validation']:
            DatasetCatalog.register(d, lambda d=d: kitti_mots_dataset(d))
            MetadataCatalog.get(d).set(thing_classes=["Pedestrian", "None", "Car"])

    kitti_mots_metadata_train = MetadataCatalog.get("kitti-mots-dataset_train")
    kitti_mots_metadata_validation = MetadataCatalog.get("kitti-mots-dataset-validation")

    # Set-up trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Train if wanted
    if TRAIN:
        print('Start training')
        trainer.train()

    # Set the dataset to test and the threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD

    # Update weights if the model has been trained
    if TRAIN:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    evaluator = COCOEvaluator("kitti-mots-dataset-validation", cfg, False, output_dir=PATH)
    val_loader = build_detection_test_loader(cfg, "kitti-mots-dataset-validation")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    print("Finishing! \n\n\n\n")

if __name__ == "__main__":

    learning_rates = [0.00075]
    epochs = [5000]
    for lr in learning_rates:
        for epoch in epochs:
            main(lr, epoch)
            REGISTERED = True

