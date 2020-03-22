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


# Resnet 50 - DC5 (lr 3x) mask_rcnn_R_50_DC5_3x
# Resnet 50-FPN (lr 3x) mask_rcnn_R_50_FPN_3x
# Resnet 101 - C4 (lr 3x) mask_rcnn_R_101_C4_3x
# R101-DC5 (lr 3x) mask_rcnn_R_101_DC5_3x
# PARAMETERS
TRAIN = True
REGISTERED = False
MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL_NAME = MODEL.split('.')[0]
LR = 0.00075
MAX_ITER = 1000
THRESHOLD = 0.6

if TRAIN:
    pkl_file_train = open('./train_KITTI-MOTS_dataset_server.pkl', 'rb')
    dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
    pkl_file_val = open('./validation_KITTI-MOTS_dataset_server.pkl', 'rb')
    dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
else:
    pkl_file_val = open('./validation_pretrained_KITTI-MOTS_dataset_server.pkl', 'rb')
    dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
    print(dataset_dicts_validation[0])

def kitti_mots_dataset(d):
    if d == "kitti-mots-dataset-train":
        return dataset_dicts_train
    else:
        return dataset_dicts_validation


def main():
    print("Starting! \n")
    print("Model: ", MODEL, "\n")
    print("LR: ", LR, "\n")
    print("MAX_ITER: ", MAX_ITER, "\n")
    print("THRESHOLD: ", THRESHOLD, "\n")

    if TRAIN:
        PATH = './output_kittimots_'+ MODEL_NAME 
    else:
        PATH = './output_kittimots_'+ MODEL_NAME  + '/TESTING'

    cfg = get_cfg()
    cfg.OUTPUT_DIR = PATH
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))

    if not TRAIN:
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        thing_classes = metadata.thing_classes
        print(thing_classes)
        del metadata # We don't need it anymore

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = LR
    cfg.SOLVER.STEPS = (1000, MAX_ITER)
    cfg.INPUT.MAX_SIZE_TRAIN = 1242
    cfg.INPUT.MAX_SIZE_TEST = 1242
    cfg.INPUT.MIN_SIZE_TRAIN = (375, )
    cfg.INPUT.MIN_SIZE_TEST = 375
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Finish configuration depending on the procedure we are performing
    if TRAIN:
        datasets = ['kitti-mots-dataset-train', 'kitti-mots-dataset-validation']
        cfg.DATASETS.TRAIN = ('kitti-mots-dataset-train',)
    else:
        datasets = ['kitti-mots-dataset-validation']
        cfg.DATASETS.TRAIN = ('kitti-mots-dataset-validation',)
    cfg.DATASETS.TEST = ("kitti-mots-dataset-validation", )

    # Register the datasets
    if not REGISTERED:
        for d in datasets:
            DatasetCatalog.register(d, lambda d=d: kitti_mots_dataset(d))
            if TRAIN:
                MetadataCatalog.get(d).set(thing_classes=["Car", "Pedestrian"])
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            else:
                MetadataCatalog.get(d).set(thing_classes=thing_classes)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

    # If training is done, get the TRAINing dataset metadata.
    """if TRAIN:
        kitti_mots_metadata_train = MetadataCatalog.get("kitti-mots-dataset_train")
    kitti_mots_metadata_validation = MetadataCatalog.get("kitti-mots-dataset-validation")"""

    # Set-up trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Train if wanted
    if TRAIN:
        print('Start training')
        trainer.train()

    # Set the dataset to test and the threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    evaluator = COCOEvaluator("kitti-mots-dataset-validation", cfg, False, output_dir=PATH)

    # Update weights if the model has been trained
    if TRAIN:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        # Start evaluation
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
    else:
        val_loader = build_detection_test_loader(cfg, "kitti-mots-dataset-validation")
        inference_on_dataset(trainer.model, val_loader, evaluator)

    print("Finishing! \n\n\n\n")

if __name__ == "__main__":

    main()
