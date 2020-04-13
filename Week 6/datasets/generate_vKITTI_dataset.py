import numpy as np
import glob
import pickle
from detectron2.structures import BoxMode
import cv2
from tqdm import tqdm


vKITTI_TO_COCO = {
    1:2
}


def generate_dataset_dicts(dataset, folders, debug=False):
    """
    Generates the dataset in COCO format

    :param dataset:
    :param folders:
    :param server:

    :return: dataset dicts
    """
    dataset_dicts_local = []
    dataset_dicts_server = []

    folder_id = 0
    for folder in folders:
        if debug:
            print(folder)

        gt_images = glob.glob(folder + '/clone/frames/instanceSegmentation/Camera_0/*.png')
        gt_images.sort()

        if debug:
            print(gt_images)

        time_frame = 0
        sequence_dicts_local = []
        sequence_dicts_server = []

        for image in tqdm(gt_images):
            mask = cv2.imread(image)
            mask = np.uint8(mask)
            color_list = []
            masks = {}
            height, width, _ = mask.shape

            for i in range(height):
                for j in range(width):
                    pixel_value = np.mean(mask[i][j][:])
                    if pixel_value not in color_list and pixel_value != 0.0:
                        color_list.append(pixel_value)
                        new_mask = np.zeros_like(mask)
                        new_mask[i][j][:] = 255
                        masks[str(pixel_value)] = new_mask
                    elif pixel_value != 0.0:
                        masks[str(pixel_value)][i][j][:] = 255
            if debug:
                print(masks.keys())

            record = {}
            record_server = {}
            record_server['file_name'] = '../mcv/datasets/' + dataset + '/' + folder.split('/')[-1] + '/clone/frames/rgb/Camera_0/rgb_' + str(time_frame).zfill(5) + '.jpg'
            record['file_name'] = '../' + dataset + '/' + folder.split('/')[-1] + '/clone/frames/rgb/Camera_0/rgb_' + str(time_frame).zfill(5) + '.jpg'
            frame_annotations = []

            for pixel_value, instance_mask in masks.items():

                class_id = 1
                height, width, _ = instance_mask.shape
                img_height = height
                img_width = width

                instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_RGB2GRAY)
                contours, _ = cv2.findContours(np.uint8(instance_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = [[int(element) for element in contour.flatten()] for contour in contours]
                segmentation = [s for s in segmentation if len(s) >= 6]
                if not segmentation:
                    continue

                x1, y1, w1, h1 = cv2.boundingRect(contours[0])
                bbox = [x1, y1, x1 + w1, y1 + h1]

                obj = {
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': vKITTI_TO_COCO[class_id],
                    'segmentation': segmentation
                }

                frame_annotations.append(obj)

            record['height'] = int(img_height)
            record['width'] = int(img_width)
            record['image_id'] = int(time_frame + int(folder_id) * 1e6)
            record['annotations'] = frame_annotations

            record_server['height'] = int(img_height)
            record_server['width'] = int(img_width)
            record_server['image_id'] = int(time_frame + int(folder_id) * 1e6)
            record_server['annotations'] = frame_annotations

            sequence_dicts_local.append(record)
            sequence_dicts_server.append(record_server)
            time_frame += 1

        if debug:
            print(len(sequence_dicts_local))
        dataset_dicts_local += sequence_dicts_local
        dataset_dicts_server += sequence_dicts_server
        folder_id += 1
    return dataset_dicts_local, dataset_dicts_server


def main():
    """
    Gets the dataset dictionaries and serializes them on a pickle file

    """

    folders = glob.glob('../vKITTI/*')
    folders.sort()

    local_train_dict, server_train_dict = generate_dataset_dicts('vKITTI', folders)

    print('Saving Pickles')
    with open('./train_synthetic_vKITTI_dataset_server.pkl', 'wb') as handle:
        pickle.dump(server_train_dict, handle)
    handle.close()

    with open('./train_synthetic_vKITTI_dataset_local.pkl', 'wb') as handle:
        pickle.dump(local_train_dict, handle)
    handle.close()


if __name__ == '__main__':
    main()
