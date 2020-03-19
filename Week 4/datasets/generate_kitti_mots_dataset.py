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


def generate_dataset_dicts(dataset, text_instances, server, type_dataset):
    """ 
    Generates the dataset in COCO format 

    :param dataset: name of the dataset (KITTI-MOTS or MOTSChallenge)
    :param text_instances: text files of the GT
    :param server: True or False depending if the pickle is for the server dataset or to work on local
    :param type_dataset: 'trained' or 'pretrained'

    :return: list of dictionaries
    
    """
    dataset_dicts = []

    for text_file in text_instances:
        print(text_file)
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
            class_id = fields[2]
            img_height = fields[3]
            img_width = fields[4]
            rle = fields[5].split('\n')[0]

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
                decode_obj = {
                    'size': [int(img_height), int(img_width)],
                    'counts': rle
                }

                bbox = toBbox(decode_obj)
                mask = decode(decode_obj)
                contours = measure.find_contours(mask, 0.5)
                del mask

                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

                if type_dataset == 'pre-trained':
                    if int(class_id) == 1:
                        class_id = 2
                    else:
                        class_id = 0
                else:
                    class_id = int(class_id) - 1

                obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                        "segmentation": [],
                        "is_crowd": False
                }
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    obj["segmentation"].append(segmentation)
                objects.append(obj)
            lines_id += 1
    return dataset_dicts


def main():
    dataset = 'KITTI-MOTS'
    #dataset = 'MOTSChallenge'
    validation = True
        
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

    local_train_dict = generate_dataset_dicts(dataset, train_text_instances, False, 'trained')
    server_train_dict = generate_dataset_dicts(dataset, train_text_instances, True, 'trained')
    if validation:
        local_val_dict = generate_dataset_dicts(dataset, val_text_instances, False, 'trained')
        server_val_dict = generate_dataset_dicts(dataset, val_text_instances, True, 'trained')
        pretrained_local_val_dict = generate_dataset_dicts(dataset, val_text_instances, False, 'pre-trained')
        pretrained_server_val_dict = generate_dataset_dicts(dataset, val_text_instances, True, 'pre-trained') 

    print('Saving Pickles')
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

        with open('./validation_pretrained_' + dataset + '_dataset_server.pkl', 'wb') as handle:
            pickle.dump(pretrained_server_val_dict, handle)
        handle.close()

        with open('./validation_pretrained_' + dataset + '_dataset_local.pkl', 'wb') as handle:
            pickle.dump(pretrained_local_val_dict, handle)
        handle.close()

if __name__ == '__main__':
    main()
