from pycocotools.coco import COCO
import pycocotools.mask as rletools
from pycocotools.mask import toBbox
from pycocotools.mask import decode
from pycocotools.mask import encode
from skimage import measure
import numpy as np
import glob
import matplotlib.pyplot as plt
import pylab
import os
import pickle
from detectron2.structures import BoxMode
import cv2


KITTIMOTS_SERVER = '/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_LOCAL = '../KITTI_MOTS/'

KITTI_TO_COCO = {
    1:2,
    2:0
}

KITTI_CLASSES = [1, 2]

def generate_dataset_dicts(dataset, text_instances, server):
    """ 
    Generates the dataset in COCO format 

    :param dataset:
    :param text_instances:
    :param server:

    :return: dataset dicts    
    """
    dataset_dicts = []
    for text_file in text_instances:
        folder = text_file.split('/')[-1].split('.')[0]

        frame_index = 0
        with open(text_file, 'r') as f:
            lines = f.readlines()
        f.close()
        images_path = glob.glob('../KITTI-MOTS/training/image_02/' + folder + '/*.png')
        images_path.sort()
        sequence_dicts = []

        for image_id in range(len(images_path)):
            frame_instances = [l for l in lines if int(l.split(' ')[0]) == image_id]
            
            if frame_instances:
                record = {}
                if server:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/training/image_02/' + folder + '/' + str(image_id).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/train/images/' + folder + '/' + str(image_id).zfill(6) + '.jpg'
                else:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../' + dataset + '/training/image_02/' + folder + '/' + str(image_id).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../' + dataset + '/train/images/' + folder + '/' + str(image_id).zfill(6) + '.jpg'
                
                record['image_id'] = image_id

                frame_annotations = []

                for instance in frame_instances:
                    fields = instance.split(' ')
                    time_frame = fields[0]
                    class_id = int(fields[2])
                    object_id = int(fields[1]) % (class_id*1000)
                    img_height = fields[3]
                    img_width = fields[4]
                    rle = fields[5].strip()

                    decode_obj = {
                    'size': [int(img_height), int(img_width)],
                    'counts': rle
                    }

                    if class_id not in KITTI_CLASSES:
                        continue

                    bbox = toBbox(decode_obj)
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

                    mask = decode(decode_obj)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = [[int(element) for element in contour.flatten()] for contour in contours]
                    segmentation = [s for s in segmentation if len(s) >= 6]
                    if not segmentation:
                        continue

                    obj = {
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': KITTI_TO_COCO[class_id],
                        'segmentation': segmentation,
                        'object_id': object_id
                    }
                    frame_annotations.append(obj)
                record['height'] = int(img_height)
                record['width'] = int(img_width)
                record['image_id'] = int(image_id + int(folder) * 1e6)
                record['annotations'] = frame_annotations
                sequence_dicts.append(record)

        print(len(sequence_dicts))
        dataset_dicts += sequence_dicts
    return dataset_dicts
            

def main():
    datasets = ['KITTI-MOTS']
    #datasets = ['vKITTI']
    local_train_dictionary = []
    server_train_dictionary = []
    local_val_dictionary = []
    server_val_dictionary = []
    local_test_dictionary = []
    server_test_dictionary = []

    for dataset in datasets:
      validation = True

      train_dataset_dicts = []
      if validation:
          val_dataset_dicts = []
          
      if dataset == 'KITTI-MOTS':
          text_instances = glob.glob('../' + dataset + '/instances_txt/*.txt')
      if dataset == 'MOTSChallenge':
          text_instances = glob.glob('../' + dataset + '/train/instances_txt/*.txt')

      text_instances.sort()

      train_text_instances = []
      val_text_instances = []
      test_text_instances = []
          
      if dataset == 'KITTI-MOTS':
          val_folders = ['0000', '0003', '0010', '0012', '0014']
          test_folders = ['0004', '0005', '0007', '0008', '0009', '0011','0015']

      if dataset == 'MOTSChallenge':
          val_folders = ['0005']
              
      for text_instance in text_instances:
          if text_instance.split('/')[-1].split('.')[0] in val_folders:
              val_text_instances.append(text_instance)
          elif text_instance.split('/')[-1].split('.')[0] in test_folders:
              test_text_instances.append(text_instance)
          else:
              train_text_instances.append(text_instance)

      local_train_dict = generate_dataset_dicts(dataset, train_text_instances, False)
      server_train_dict = generate_dataset_dicts(dataset, train_text_instances, True)
      local_val_dict = generate_dataset_dicts(dataset, val_text_instances, False)
      server_val_dict = generate_dataset_dicts(dataset, val_text_instances, True)
      local_test_dict = generate_dataset_dicts(dataset, test_text_instances, False)
      server_test_dict = generate_dataset_dicts(dataset, test_text_instances, True)

      local_train_dictionary += local_train_dict
      server_train_dictionary += server_train_dict
      local_val_dictionary += local_val_dict
      server_val_dictionary += server_val_dict
      local_test_dictionary += local_test_dict
      server_test_dictionary += server_test_dict

    
    print('Saving Pickles')
    with open('./train_real_' + datasets[0] + '_dataset_server.pkl', 'wb') as handle:
        pickle.dump(server_train_dictionary, handle)
    handle.close()

    with open('./train_real_'+ datasets[0] + '_dataset_local.pkl', 'wb') as handle:
        pickle.dump(local_train_dictionary, handle)
    handle.close()

    with open('./validation_real_' + datasets[0] + '_dataset_server.pkl', 'wb') as handle:
        pickle.dump(server_val_dictionary, handle)
    handle.close()

    with open('./validation_real_' + datasets[0] + '_dataset_local.pkl', 'wb') as handle:
        pickle.dump(local_val_dictionary, handle)
    handle.close()

    with open('./test_real_' + datasets[0] + '_dataset_server.pkl', 'wb') as handle:
        pickle.dump(server_test_dictionary, handle)
    handle.close()

    with open('./test_real_' + datasets[0] + '_dataset_local.pkl', 'wb') as handle:
        pickle.dump(local_test_dictionary, handle)
    handle.close()


main()