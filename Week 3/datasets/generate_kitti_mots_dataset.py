from pycocotools.coco import COCO
import pycocotools.mask as rletools
from pycocotools.mask import toBbox
from pycocotools.mask import decode
from pycocotools.mask import encode
import numpy as np
import glob
import matplotlib.pyplot as plt
import pylab
import os
import pickle
from detectron2.structures import BoxMode

from pycocotools.coco import COCO
import pycocotools.mask as rletools
from pycocotools.mask import toBbox
from pycocotools.mask import decode
from pycocotools.mask import encode
import numpy as np
import glob
import matplotlib.pyplot as plt
import pylab
import os
import pickle
from detectron2.structures import BoxMode

def generate_dataset_dicts(dataset, text_instances, server):
    """ Generates the dataset in COCO format """
    dataset_dicts = []
    dataset_dicts = []

    for text_file in text_instances:
        folder = text_file.split('/')[-1]
        folder = folder.split('.')[0]

        frame_index = 0
        with open(text_file, 'r') as f:
            lines = f.readlines()
        f.close()
        objects = []
        record = {}
        
        lines_id = 0
        for line in lines:
            fields = line.split(' ')
            time_frame = fields[0]
            object_id = fields[1]
            class_id = fields[2]
            img_height = fields[3]
            img_width = fields[4]
            rle = fields[5].split('\n')[0]

            decode_obj = {'size': [int(img_height), int(img_width)],
                        'counts': rle
            }
            bbox = toBbox(decode_obj)
            if lines_id == 0:
                if server:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/training/image_02/' + folder + '/' + str(time_frame).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/train/images/' + folder + '/' + str(time_frame).zfill(6) + '.jpg'
                else:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../' + dataset + '/training/image_02/' + folder + '/' +str(time_frame).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../' + dataset + '/train/images/' + folder + '/' +str(time_frame).zfill(6) + '.jpg'
                            
                record['image_id'] = frame_index
                record['height'] = int(img_height)
                record['width'] = int(img_width)

            if int(time_frame) != frame_index:
                record["annotations"] = objects
                dataset_dicts.append(record)

                frame_index += 1
                record = {}
                objects = []

                if server:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/training/image_02/' + folder + '/' + str(frame_index).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../mcv/datasets/' + dataset + '/train/images/' + folder + '/' + str(frame_index).zfill(6) + '.jpg'

                else:
                    if dataset == 'KITTI-MOTS':
                        record['file_name'] = '../' + dataset + '/training/image_02/' + folder + '/' +str(frame_index).zfill(6) + '.png'
                    if dataset == 'MOTSChallenge':
                        record['file_name'] = '../' + dataset + '/train/images/' + folder + '/' +str(frame_index).zfill(6) + '.jpg'
                        
                record['image_id'] = frame_index
                record['height'] = int(img_height)
                record['width'] = int(img_width)
            
            if int(class_id) != 10:
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                if int(class_id) == 1:
                    class_id = 2
                elif int(class_id) == 2:
                    class_id = 0
                obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(class_id)
                }
                objects.append(obj)
            lines_id += 1
    #print(dataset_dicts)
    return dataset_dicts


def main():
    #dataset = 'KITTI-MOTS'
    dataset = 'MOTSChallenge'
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

    if validation:
        val_text_instances = []
        
        if dataset == 'KITTI-MOTS':
            val_folders = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
        if dataset == 'MOTSChallenge':
            val_folders = ['0005']
            
        for text_instance in text_instances:
            if text_instance.split('/')[-1].split('.')[0] in val_folders:
                # print(text_instance)
                val_text_instances.append(text_instance)
            else:
                train_text_instances.append(text_instance)
    else:
        train_text_instances.append(text_instance)

    local_train_dict = generate_dataset_dicts(dataset, train_text_instances, False)
    server_train_dict = generate_dataset_dicts(dataset, train_text_instances, True)
    if validation:
        local_val_dict = generate_dataset_dicts(dataset, val_text_instances, False)
        server_val_dict = generate_dataset_dicts(dataset, val_text_instances, True)

    with open('./train_' + dataset + '_dataset_server.pkl', 'wb') as handle:
        pickle.dump(server_train_dict, handle)
    handle.close()

    with open('./train_' + dataset + '_dataset_local.pkl', 'wb') as handle:
        pickle.dump(local_train_dict, handle)
    handle.close()

    if validation:
        with open('./validation_' + dataset + '_dataset_server.pkl', 'wb') as handle:
            pickle.dump(server_val_dict, handle)
        handle.close()

        with open('./validation_' + dataset + '_dataset_local.pkl', 'wb') as handle:
            pickle.dump(local_val_dict, handle)
        handle.close()

if __name__ == '__main__':
    main()
