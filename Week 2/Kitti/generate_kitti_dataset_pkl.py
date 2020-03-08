"""

This module contains the functionality to load images from the dataset,
alongside the GT for their use with Facebook's AI detectron2.

"""

import glob
import pickle

import cv2
from detectron2.structures import BoxMode


def get_kitti_dataset(img_dir, labels_dir, debug=False):
    """
    This function loads the info of the KITTI Dataset
    """

    categories = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8
    }

    images_files = glob.glob(img_dir + '/*.png')
    images_files.sort()
    labels_files = glob.glob(labels_dir + '/*.txt')
    labels_files.sort()

    if debug:
        print(images_files)
        print(labels_files)
        print('Nr. of images: ', len(images_files))
        print('Nr. of labels: ', len(labels_files))

    # Otherwise something is wrong
    if len(images_files) == len(labels_files):

        dataset_dicts = []
        for i in range(len(labels_files)):
            if debug:
                print("Annotating image ", i)
            record = {}
            height, width = cv2.imread(images_files[i]).shape[:2]
            image = cv2.imread(images_files[i])

            record['file_name'] = images_files[i]
            record['image_id'] = i
            record['height'] = height
            record['width'] = width

            with open(labels_files[i], "r") as labels:
                lines = labels.readlines()
                labels.close()

            objects = []
            for line in lines:
                elements = line.split(' ')

                if debug:
                    print(elements)

                obj = {
                    "bbox": [float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": categories[elements[0]]
                }

                if debug:
                    cv2.rectangle(image, (int(float(elements[4])), int(float(elements[5]))), (int(float(elements[6])), int(float(elements[7]))), (255,255,255))
                    cv2.putText(image, elements[0], (int(float(elements[4])), int(float(elements[5]))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA) 
                objects.append(obj)

            record["annotations"] = objects
            dataset_dicts.append(record)

        with open('./kitti_dataset.pkl', 'wb') as handle:
            pickle.dump(dataset_dicts, handle)

        return dataset_dicts


# For testing purposes
if __name__ == '__main__':
    TRAIN_PATH_IMAGES = '../KITTI/data_object_image_2/training/image_2'
    TRAIN_LABELS_IMAGES = '../KITTI/training/label_2'
    TEST_PATH_IMAGES = '../MIT_split/test/'

    _ = get_kitti_dataset(TRAIN_PATH_IMAGES, TRAIN_LABELS_IMAGES, debug=True)
