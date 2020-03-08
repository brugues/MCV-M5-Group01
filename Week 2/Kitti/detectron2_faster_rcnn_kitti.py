# import some common libraries
import glob
import os
import pickle

import cv2

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Dataset guardado en pkl file. El script para generar el pickle está en GitHub
pkl_file = open('./kitti_dataset.pkl', 'rb')
dataset_dicts = pickle.load(pkl_file, fix_imports=True, encoding='ASCII', errors='strict')

def kitti_dataset(path):
    return dataset_dicts

TRAIN_PATH_IMAGES = '../mcv/datasets/KITTI/data_object_image_2/training/image_2'
TRAIN_LABELS_IMAGES = '../mcv/datasets/KITTI/training/label_2'

for d in ["kitti_dataset"]:
    DatasetCatalog.register(d, lambda d=d: kitti_dataset(d))
    MetadataCatalog.get(d).set(thing_classes=["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"])

kitti_metadata = MetadataCatalog.get("kitti_dataset")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kitti_dataset",)
cfg.DATASETS.TEST = ("kitti_dataset",)
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

from detectron2.checkpoint import DetectionCheckpointer

cfg.MODEL.WEIGHTS = './output/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

images = glob.glob(TRAIN_PATH_IMAGES + '/*.png')
images.sort()

categories = {
        0: 'Car',
        1: 'Van',
        2: 'Truck',
        3: 'Pedestrian',
        4: 'Person_sitting',
        5: 'Cyclist',
        6: 'Tram',
        7: 'Misc',
        8: 'DontCare',
    }

PATH = './output/training_results'

if not os.path.exists(PATH):
    os.makedirs(PATH)

image_id = 0
for image in images:
    print('Writing file ', image_id)
    im = cv2.imread(image)

    # Make predictions
    outputs = predictor(im)

    instances = outputs['instances'].to('cpu'); instances = instances.get_fields()
    bboxes = instances['pred_boxes']; bboxes = bboxes.tensor.tolist()
    predicted_classes = instances['pred_classes']; predicted_classes.tolist()
    confidences = instances['scores']; confidences.tolist()

    filename = image.split('/')[-1].split('.')[0] + '.txt'
    file = open(os.path.join(PATH, filename), 'w')

    for i in range(len(bboxes)):
        predicted_class = predicted_classes[i]
        bbox = bboxes[i]

        element_class = categories[int(predicted_class)]
        confidence = str(float(confidences[i]))
        bbox_low_x = str(bbox[0]) + ' '
        bbox_low_y = str(bbox[1]) + ' '
        bbox_high_x = str(bbox[2]) + ' '
        bbox_high_y = str(bbox[3]) + ' '

        line = element_class + ' -1 -1 -10 '
        line = line + bbox_low_x + bbox_low_y + bbox_high_x + bbox_high_y
        line = line + '-1 -1 -1 -1000 -1000 -1000 -10 '
        line = line + confidence + '\n'

        file.write(line)

    file.close()
    image_id += 1
