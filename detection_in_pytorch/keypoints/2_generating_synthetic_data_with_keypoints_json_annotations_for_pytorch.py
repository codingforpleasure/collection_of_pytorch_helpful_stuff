# Some of the logic was taken from here:
# https://towardsdatascience.com/geometric-transformations-in-computer-vision-an-intuitive-explanation-with-python-examples-b0b6f06e1844
# IT works well!

# Gil Pay Attention:
# 1. The size of the background.
# 2. The size of the object should fit at least into the background in both dimensions.


# Assumptions the INPUT csv is in the following format:

# filename,keypoint0-x,keypoint0-y,keypoint1-x,keypoint1-y,width,height
# match1.png,21.0,3.0,11.0,296.0,32,300
# match4.png,14.0,1.0,12.0,298.0,26,300

# * This format was generated using the script: json_labelme_to_csv.py

import glob
import random
import shutil

import cv2
import os
import numpy as np
import pandas as pd
from random import randint
from typing import Iterable, Any, Dict  # List, Tuple, etc...
from itertools import product
import json


# The function was taken from her:
# https://github.com/cvzone/cvzone/blob/master/cvzone/Utils.py

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    height_front, width_front, channels_front = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)

    imgMaskFull[pos[1]:height_front + pos[1], pos[0]:width_front + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:height_front + pos[1], pos[0]:width_front + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


def rotate_object_img(input_img):
    center = (input_img.shape[1] // 2, input_img.shape[0] // 2)  # Get the image center
    angle = random.randint(-180, 180)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)  # Calculate the rotation matrix

    original_four_corners = [(0, 0, 1), (input_img.shape[1], 0, 1), (0, input_img.shape[0], 1),
                             (input_img.shape[1], input_img.shape[0], 1)]
    new_corners = [np.matmul(rotation_matrix, pt) for pt in original_four_corners]
    min_x = np.min([pt[0] for pt in new_corners])
    max_x = np.max([pt[0] for pt in new_corners])
    min_y = np.min([pt[1] for pt in new_corners])
    max_y = np.max([pt[1] for pt in new_corners])
    new_dimensions = (int(max_y - min_y), int(max_x - min_x))

    new_center = (new_dimensions[1] // 2, new_dimensions[0] // 2)
    center_translation = (new_center[0] - center[0], new_center[1] - center[1])
    rotation_matrix[0][2] += center_translation[0]
    rotation_matrix[1][2] += center_translation[1]
    rotated_img = cv2.warpAffine(input_img, rotation_matrix, (new_dimensions[1], new_dimensions[0]))
    return rotated_img, rotation_matrix


def get_rotated_keypoints(original_keypoints_on_objects, filename):
    relevant_columns = original_keypoints_on_objects.columns.drop(['width', 'height'])
    coordinates_object = dict(original_keypoints_on_objects.loc[filename, relevant_columns])

    collection_of_original_keypoints = np.array([list(coordinates_object.values())]).reshape(num_keypoints_per_img, -1)

    collection_of_original_keypoints = np.column_stack(
        (collection_of_original_keypoints, np.ones((num_keypoints_per_img, 1))))

    new_keypoints = [np.matmul(rotation_matrix, pt) for pt in collection_of_original_keypoints]

    return new_keypoints


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


def remove_directory_content(dir_path):
    for files in os.listdir(dir_path):
        path = os.path.join(dir_path, files)
        os.remove(path)


def clean_all_directories():
    dir_types = ['train', 'valid', 'test']
    for dir_type in dir_types:
        dir_img = f'./generated_matches/{dir_type}/images'
        dir_annotations = f'./generated_matches/{dir_type}/annotations'
        remove_directory_content(dir_img)
        remove_directory_content(dir_annotations)


if __name__ == '__main__':

    # clean_all_directories() # Both 'train' and 'valid' and 'test'

    debug_mode_show_keypoints = False
    debug_mode_show_bbox = False

    num_objects_on_page = 2

    directory_type = 'test'  # Either train/valid/test/
    dataset_generated_output_dir_images = f'./generated_matches/{directory_type}/images'
    dataset_generated_output_dir_annotations = f'./generated_matches/{directory_type}/annotations'

    all_bg_files = glob.glob("imgs_background/*.png")

    all_object_files = [glob.glob("imgs_single_object/*.png")[1]]

    input_keypoints_on_object = pd.read_csv('imgs_single_object/single_object_keypoints.csv', index_col='filename')

    # The -2 is because the two last columns are width and height
    num_keypoints_per_img = int((len(input_keypoints_on_object.columns) - 2) / 2)

    columns_name_output_csv = ['filename'] + list(input_keypoints_on_object.columns[:-2]) + ['bbox-width',
                                                                                             'bbox-height',
                                                                                             'bbox-uppercorner-x',
                                                                                             'bbox-uppercorner-y']

    filenames = []

    parameters = {"iter": list(range(200)),
                  "filename_full_path_to_bg": all_bg_files,
                  "filename_full_path_to_object": all_object_files}  # "match_idx": list(range(num_objects_on_page))

    for iter, settings in enumerate(grid_parameters(parameters)):
        print("iter: ", iter)
        bg_filename = os.path.basename(settings["filename_full_path_to_bg"])
        img_bg = cv2.imread(settings["filename_full_path_to_bg"], cv2.IMREAD_COLOR)

        img_bg_backup = img_bg.copy()

        img_object = cv2.imread(settings["filename_full_path_to_object"], cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(settings["filename_full_path_to_object"])
        # print("Focusing on match: ", filename)
        exported_img_filename_with_bg = filename[:-4] + '_keypoint_rotated_with_' + bg_filename[:-4] + '_' + \
                                        str(settings["iter"]) + '.jpeg'

        exported_json_filename = exported_img_filename_with_bg[:-4] + 'json'

        img_bg = img_bg_backup.copy()
        bboxes_per_img = []
        keypoints_per_img = []
        for idx in range(num_objects_on_page):
            rotated_keypoints_per_object = [exported_img_filename_with_bg]
            rotated_object_output, rotation_matrix = rotate_object_img(img_object)

            rotated_keypoints = get_rotated_keypoints(input_keypoints_on_object, filename)

            # For debugging print the actual keypoints after
            if debug_mode_show_keypoints:
                for rotated_point in rotated_keypoints:
                    cv2.circle(rotated_object_output,
                               center=(int(rotated_point[0]), int(rotated_point[1])),
                               radius=5,
                               color=(0, 0, 0, 255),
                               thickness=-1)

            x_margin = img_bg.shape[1] - rotated_object_output.shape[1] - 1
            # print('x_margin: ', x_margin)
            x_offset = randint(0, x_margin)
            y_margin = img_bg.shape[0] - rotated_object_output.shape[0] - 1
            # print('y_margin: ', y_margin)
            y_offset = randint(0, y_margin)
            # print('x_offset %s, y_offset: %s' % (x_offset, y_offset))
            # print('x_offset %s, y_offset: %s' % (x_offset, y_offset))

            final_image = overlayPNG(imgBack=img_bg,
                                     imgFront=rotated_object_output,
                                     pos=[x_offset, y_offset])

            img_bg = final_image.copy()

            keypoints_per_object = []
            for rotated_point in rotated_keypoints:

                keypoint = [int(rotated_point[0] + x_offset), int(rotated_point[1] + y_offset), 1]
                keypoints_per_object.append(keypoint)

                if debug_mode_show_keypoints:
                    cv2.circle(img_bg,
                               center=(int(rotated_point[0] + x_offset), int(rotated_point[1] + y_offset)),
                               radius=5,
                               color=(0, 0, 0, 255),
                               thickness=-1)

            keypoints_per_img.append(keypoints_per_object)

            bboxes_per_img.append([int(x_offset),  # bbox-uppercorner-x
                                   int(y_offset),  # bbox-uppercorner-y
                                   int(x_offset + rotated_object_output.shape[1]),  # bbox-lowerrcorner-x
                                   int(y_offset + rotated_object_output.shape[0])  # bbox-lowerrcorner-y
                                   ])

            if debug_mode_show_bbox:
                cv2.rectangle(img_bg,
                              pt1=(x_offset, y_offset),
                              pt2=(
                                  x_offset + rotated_object_output.shape[1],
                                  y_offset + rotated_object_output.shape[0]),
                              color=(255, 0, 0),
                              thickness=2)

            cv2.imwrite(os.path.join(dataset_generated_output_dir_images, exported_img_filename_with_bg), img_bg)

        # Writing the json file per image
        json_body = {
            "bboxes": bboxes_per_img,
            "keypoints": keypoints_per_img
        }

        with open(os.path.join(dataset_generated_output_dir_annotations, exported_json_filename), 'w') as fd:
            json.dump(json_body, fd, indent=2, ensure_ascii=False)

    print("\nAll generated images appear in directory: ", dataset_generated_output_dir_images)
