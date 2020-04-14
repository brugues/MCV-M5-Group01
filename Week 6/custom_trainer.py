import copy
import numpy as np
import torch
import cv2

from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.data import detection_utils as utils



class CTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomMapper(cfg, True))

class CustomMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.dataset_dict = copy.deepcopy(cfg)

        self.flip = MetadataCatalog.get(self.dataset_dict.DATASETS.TRAIN[0]).get('flip')
        self.crop = MetadataCatalog.get(self.dataset_dict.DATASETS.TRAIN[0]).get('crop')
        self.saturation = MetadataCatalog.get(self.dataset_dict.DATASETS.TRAIN[0]).get('saturation')
        self.rotation = MetadataCatalog.get(self.dataset_dict.DATASETS.TRAIN[0]).get('rotation')

        self.crop_type = cfg.INPUT.CROP.TYPE
        self.crop_size = cfg.INPUT.CROP.SIZE


    def __call__(self, dataset_dict):
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)
        data_transformations = []

        if self.is_train:
            # Crop
            if self.crop:
                crop_gen = T.RandomCrop(self.crop_type, self.crop_size)
                data_transformations.append(crop_gen)
                print('crop')
            # Horizontal flip
            if self.flip:
                flip_gen = T.RandomFlip()
                data_transformations.append(flip_gen)
            # if self.rotation:
            #     rotation_gen = T.RandomRotation([0, 90])
            #     data_transformations.append(rotation_gen)
            if self.saturation:
                saturation_gen = T.RandomSaturation(0.5, 1.5)
                data_transformations.append(saturation_gen)
                print(str(dataset_dict["file_name"]))

        image, transforms = T.apply_transform_gens(data_transformations, image)
        print('\n\n -------------------PRINTING IMAGE---------------------- \n\n')
        img_name = dataset_dict["file_name"][len(dataset_dict["file_name"])-15:len(dataset_dict["file_name"])-4]
        img_name = '/home/grupo01/images_augmented/' + img_name +'_augmented.png'
        print(len(dataset_dict["file_name"]))
        print(img_name)

        cv2.imwrite(img_name, image)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        image_shape = image.shape[:2]

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)


            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
