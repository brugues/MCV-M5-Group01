import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob
import os
import pickle
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from tracking import print_rect, MultiTracker
from mot_evaluation import mot_metrics, get_centroid

# KITTI-MOTS TEST IMAGES
TEST_PATH = '../KITTI-MOTS/training/image_02/'
images_1 = glob.glob(TEST_PATH + '0004/*')
images_2 = glob.glob(TEST_PATH + '0005/*')
images_3 = glob.glob(TEST_PATH + '0007/*')
images_4 = glob.glob(TEST_PATH + '0008/*')
images_5 = glob.glob(TEST_PATH + '0009/*')
images_6 = glob.glob(TEST_PATH + '0011/*')
images_7 = glob.glob(TEST_PATH + '0015/*')

sequences = [images_1, images_2, images_3, images_4, images_5, images_6, images_7]

# Create config
network = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
path = './Week6/optional'
os.makedirs(path, exist_ok=True)

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(network))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(network)
#cfg.MODEL.WEIGHTS = os.path.join('./Week5/models/Maria/mask_rcnn_X_101_32x8d_FPN_3x'+ network.split('.')[0], "EPOCHS_/model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join('./Week6/optional', 'model_final.pth')
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

pkl_file_train = open('../KITTI-MOTS/train_real_KITTI-MOTS_dataset_local.pkl', 'rb')
dataset_dicts_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
pkl_file_val = open('../KITTI-MOTS/test_real_KITTI-MOTS_dataset_local.pkl', 'rb')
dataset_dicts_validation = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')

print(dataset_dicts_validation)
datasets = ['kitti-mots-dataset-train', 'kitti-mots-dataset-validation']
cfg.DATASETS.TRAIN = ('kitti-mots-dataset-validation',)

def kitti_mots_dataset(d):
    if d == "kitti-mots-dataset-train":
        return dataset_dicts_train
    else:
        return dataset_dicts_validation

# Register the datasets
for d in datasets:
    DatasetCatalog.register(d, lambda d=d: kitti_mots_dataset(d))
    MetadataCatalog.get(d).set(thing_classes=["Person", "None", "Car"])

cfg.DATASETS.TEST = ('kitti-mots-dataset-validation',)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
    
idx = 0
metadata_val = MetadataCatalog.get("kitti-mots-dataset-validation")
tracker = MultiTracker(ttype='overlap', key=get_centroid)
predictor = DefaultPredictor(cfg)
tracking_metrics = mot_metrics()

start_sequence = len(sequences[0])
end_sequence = len(sequences[0]) + len(sequences[1])
out_cap = None
print('start: ', start_sequence) 
print('end: ', end_sequence)
for entry in dataset_dicts_validation:
  if idx > start_sequence and idx < end_sequence:
    det_boxes = []
    gt_rects = []
    dt_rects = {}
    dt_track = OrderedDict()

    img = cv2.imread(entry["file_name"])
    # Make prediction
    outputs = predictor(img)
    pred_boxes = outputs["instances"].to("cpu").get_fields()["pred_boxes"].tensor.tolist()
    
    for box in pred_boxes:
      det_boxes.append(box)
    dt_rects = tracker.update(det_boxes)
    
    for box in entry["annotations"]:
      box_and_id = []
      gt_id = int(box["object_id"])
      box = box["bbox"]
      for coord in range(len(box)):
        box[coord] = int(box[coord])
        box_and_id.append(box[coord])
      box_and_id.append(gt_id)
      box_and_id.append(0)
      gt_rects.append(box_and_id)

    for dt_id, dtrect in dt_rects.items():
      dt_track.update({dt_id: tracker.object_paths[dt_id]})
      print_rect(img, dtrect,  (0, 255, 0), dt_id)
    
    tracking_metrics.update(dt_track,gt_rects)
    if out_cap is None:
      fshape = img.shape
      out_cap = cv2.VideoWriter(path + '/sequence05.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (fshape[1],fshape[0]))

    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_val, scale=1.2)
    vis = visualizer.draw_dataset_dict(entry)
    # cv2_imshow(vis.get_image()[:, :, ::-1])
    # cv2.imwrite(path + '/tracking.png', vis.get_image()[:, :, ::-1])
    # out_cap.write(img.astype('uint8'))

  idx += 1
  print(idx)
    
out_cap.release()

idf1, idp, idr, mota = tracking_metrics.get_metrics()
print("IDF1: ", idf1)
print("MOTA: ", mota)
    
