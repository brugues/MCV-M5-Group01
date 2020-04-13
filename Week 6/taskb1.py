# import some common libraries
import pickle
import os
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()

# NETWORK PARAMETERS
LR = 0.001
MAX_ITER = 1000
THRESHOLD = 0.6
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0001
MIN_STEP = 500
MIN_SIZE_TRAIN = 800
MAX_SIZE_TRAIN = 2500
MIN_SIZE_TEST = 800 
MAX_SIZE_TEST = 2500

# OTHER PARAMETERS
TRAIN = True
BASE_PATH = 'output_taskb_pep/'
MODEL = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
MODEL_NAME = MODEL.split('.')[0]
MODEL_PRETRAIN_TYPE = MODEL_NAME.split('/')[0]

DATASETS = ['test_real_KITTI-MOTS_dataset_server.pkl',
            'train_real_KITTI-MOTS_dataset_server.pkl',
            'validation_real_KITTI-MOTS_dataset_server.pkl',
            'train_synthetic_vKITTI_dataset_server.pkl']
DATASETS_DICTS = {}

RESULTS = {}


def kitti_mots_dataset(d):
    """ Returns the pickle file for the given dataset string """

    global DATASETS_DICTS
    return DATASETS_DICTS[d]


def register_datasets():
    """ Registers all the available datasets"""

    global DATASETS_DICTS, DATASETS

    for dataset in DATASETS:
        pkl_file = open('./pickles/' + dataset, 'rb')
        DATASETS_DICTS[dataset] = pickle.load(pkl_file, fix_imports=True, encoding='ASCII', errors='strict')
        pkl_file.close()

    for dataset in DATASETS:
        DatasetCatalog.register(dataset, lambda dataset=dataset: kitti_mots_dataset(dataset))
        MetadataCatalog.get(dataset).set(thing_classes=["Pedestrian", "None", "Car"])


def train(train_datasets, validation_dataset, output_path):
    global LR, MAX_ITER, THRESHOLD, MAX_ITER, MODEL_NAME

    print("Starting! \n")
    print("Model: ", MODEL, "\n")
    print("LR: ", LR, "\n")
    print("MAX_ITER: ", MAX_ITER, "\n")
    print("THRESHOLD: ", THRESHOLD, "\n")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = LR
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.MOMENTUM = MOMENTUM
    cfg.SOLVER.WEIGHT_DECAY = WEIGHT_DECAY
    cfg.SOLVER.STEPS = (MIN_STEP, MAX_ITER)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = validation_dataset

    # Set-up trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Train if wanted
    print('Start training')
    trainer.train()

    # Update weights and threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    evaluator = COCOEvaluator(validation_dataset[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, validation_dataset[0])
    inference_result = inference_on_dataset(trainer.model, val_loader, evaluator)

    print("Finishing evaluation after training \n\n")
    return inference_result, cfg


def test(test_dataset, config):
    """
        Perfoms evaluation on the dataset using the model on the config parameter.
    """

    global MODEL

    config.DATASETS.TEST = test_dataset

    # Set-up trainer
    trainer = DefaultTrainer(config)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator(test_dataset[0], config, False, output_dir=config.OUTPUT_DIR + '/test')
    val_loader = build_detection_test_loader(config, test_dataset[0])
    return inference_on_dataset(trainer.model, val_loader, evaluator)


def run_training(subtask, dataset_train, dataset_val):
    """
        Calls the training function and stores the results for later use
    """

    global RESULTS, BASE_PATH, MODEL_PRETRAIN_TYPE, LR, MAX_ITER

    experiment = {}
    experiment['type'] = 'TRAIN'
    experiment['lr'] = LR
    experiment['max_iter'] = MAX_ITER
    experiment['output_path'] = BASE_PATH + 'TASK_' + subtask + '_' + MODEL_PRETRAIN_TYPE + '_LR_'  + str(LR) + '_ITER_' + str(MAX_ITER)

    result, cfg = train(dataset_train, dataset_val, experiment['output_path'])

    results = dict(result)

    experiment['full_result'] = result
    experiment['AP'] = float(results['bbox']['AP'])
    experiment['cfg'] = cfg
    return experiment


def run_testing(dataset_test, subtask):
    """
        Calls the testing function and stores the results in a file
    """

    global RESULTS, BASE_PATH

    idx = 1
    best_score = 0.0
    idx_best_score = 1

    for _, experiment in RESULTS[subtask].items():
        if experiment['AP'] > best_score:
            best_score = experiment['AP']
            idx_best_score = idx
        idx += 1

    experiment = {}
    experiment['type'] = 'TEST'
    experiment['lr'] = RESULTS[subtask][str(idx_best_score)]['lr']
    experiment['max_iter'] = RESULTS[subtask][str(idx_best_score)]['max_iter']
    experiment['cfg'] = RESULTS[subtask][str(idx_best_score)]['cfg']
    experiment['full_result'] = test(dataset_test, experiment['cfg'])

    experiment['cfg'] = ''
    with open(BASE_PATH + 'task_' + subtask + '_best_result.json', 'w') as test_file:
        test_file.write(json.dumps(experiment))
    test_file.close()


def main():
    """
        Configures the experiments and runs them
    """

    global LR, MAX_ITER, RESULTS, MODEL, MODEL_NAME, MODEL_PRETRAIN_TYPE, BASE_PATH

    learning_rates = [0.00025, 0.00075]
    epochs = [3000]
    models = ["Cityscapes/mask_rcnn_R_50_FPN.yaml"]

    register_datasets()

    # Configure tasks
    tasks = ['B1_1', 'B1_2','B1_3']
    tasks_parameters = {
        tasks[0] : {
            'train' : (DATASETS[1],),
            'validation': (DATASETS[2],),
            'test': (DATASETS[0],)
        },
        tasks[1] : {
            'train' : (DATASETS[3],),
            'validation': (DATASETS[2],),
            'test': (DATASETS[0],)
        },
        tasks[2] : {
            'train' : (DATASETS[3], DATASETS[1],),
            'validation': (DATASETS[2],),
            'test': (DATASETS[0],)
        }
    }

    for task in tasks:
        RESULTS[task] = {}
        idx = 1
        for lr in learning_rates:
            LR = lr
            for epoch in epochs:
                MAX_ITER = epoch
                for model in models:
                    MODEL = model
                    MODEL_NAME = MODEL.split('.')[0]
                    MODEL_PRETRAIN_TYPE = MODEL_NAME.split('/')[0]
                    RESULTS[task][str(idx)] = run_training(task,
                                                           tasks_parameters[task]['train'],
                                                           tasks_parameters[task]['validation'])
                    idx += 1

        with open(BASE_PATH + 'task_' + task + '_crossval_results.json', 'w') as file:
            file.write(json.dumps(RESULTS[task]))
        file.close()

        run_testing(tasks_parameters[task]['test'], task)


if __name__ == "__main__":
    main()
