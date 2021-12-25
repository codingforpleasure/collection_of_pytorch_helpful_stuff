# Some of the logic was taken from here:
# https://towardsdatascience.com/geometric-transformations-in-computer-vision-an-intuitive-explanation-with-python-examples-b0b6f06e1844
# IT works well!

import glob
import cv2
import os
import numpy as np
import pandas as pd
from random import randint


# The function was taken from her:
# https://github.com/cvzone/cvzone/blob/master/cvzone/Utils.py

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


if __name__ == '__main__':
    debug_mode = False
    img_bg = cv2.imread(
        "/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/backgrounds/background8.jpg",
        cv2.IMREAD_COLOR)

    all_object_files = glob.glob(
        "/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/matches/*.png")

    original_keypoints_on_objects = pd.read_csv('./matches/matches_keypoints.csv', index_col='filename')

    filenames = []

    all_keypoints_per_all_imgs = []

    for filename_full_path in all_object_files:

        new_keypoints_per_background_img = []

        img_object = cv2.imread(filename_full_path, cv2.IMREAD_UNCHANGED)

        filename = os.path.basename(filename_full_path)
        exported_file_with_bg = filename[:-4] + '_keypoint_rotated_with_bg.jpeg'
        new_keypoints_per_background_img.append(exported_file_with_bg)

        center = (img_object.shape[1] // 2, img_object.shape[0] // 2)  # Get the image center
        rotation_matrix = cv2.getRotationMatrix2D(center, -45, 1)  # Calculate the rotation matrix
        new_img = cv2.warpAffine(img_object, rotation_matrix, (img_object.shape[1], img_object.shape[0]))

        original_four_corners = [(0, 0, 1), (img_object.shape[1], 0, 1), (0, img_object.shape[0], 1),
                                 (img_object.shape[1], img_object.shape[0], 1)]
        new_corners = [np.matmul(rotation_matrix, pt) for pt in original_four_corners]

        min_x = np.min([pt[0] for pt in new_corners])
        max_x = np.max([pt[0] for pt in new_corners])

        min_y = np.min([pt[1] for pt in new_corners])
        max_y = np.max([pt[1] for pt in new_corners])

        new_dimensions = (int(max_y - min_y), int(max_x - min_x))
        print("Dimesnions of our new image should be: ", new_dimensions)

        new_center = (new_dimensions[0] // 2, new_dimensions[1] // 2)
        center_translation = (new_center[0] - center[0], new_center[1] - center[1])

        rotation_matrix[0][2] += center_translation[0]
        rotation_matrix[1][2] += center_translation[1]

        new_img2 = cv2.warpAffine(img_object, rotation_matrix, new_dimensions)

        # cv2.imwrite("/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/example.png", new_img2)

        coordinates_object = dict(original_keypoints_on_objects.loc[filename,])

        collection_of_original_keypoints = np.array([
            [coordinates_object['keypoint1-x'], coordinates_object['keypoint1-y'], 1],
            [coordinates_object['keypoint2-x'], coordinates_object['keypoint2-y'], 1],
        ])

        new_keypoints = [np.matmul(rotation_matrix, pt) for pt in collection_of_original_keypoints]

        # For debugging print the actual keypoints after
        if debug_mode:
            for rotated_point in new_keypoints:
                # Convert to homogenous coordinates in np array format first so that you can pre-multiply M
                cv2.circle(new_img2, (int(rotated_point[0]), int(rotated_point[1])), 30, (0, 0, 0, 255), -1)

        # cv2.imwrite(filename[:-4] + '_keypoint_rotated.png', new_img2)
        x_margin = img_bg.shape[1] - new_img2.shape[1]
        x_offset = randint(0, x_margin)

        y_margin = img_bg.shape[0] - new_img2.shape[0]
        y_offset = randint(0, y_margin)

        final_image = overlayPNG(imgBack=img_bg,
                                 imgFront=new_img2,
                                 pos=[x_offset, y_offset])

        for rotated_point in new_keypoints:
            # Convert to homogenous coordinates in np array format first so that
            # you can pre-multiply M
            if debug_mode:
                cv2.circle(final_image,
                           (int(rotated_point[0] + x_offset), int(rotated_point[1] + y_offset)),
                           radius=30,
                           color=(0, 0, 0, 255),
                           thickness=-1)

            new_keypoints_per_background_img.append(int(rotated_point[0] + x_offset))
            new_keypoints_per_background_img.append(int(rotated_point[1] + y_offset))

        all_keypoints_per_all_imgs.append(new_keypoints_per_background_img)

        cv2.imwrite(exported_file_with_bg, final_image)

    df_keypoints_output = pd.DataFrame(all_keypoints_per_all_imgs,
                      columns=['filename']+list(original_keypoints_on_objects.columns))

    df_keypoints_output.to_csv('keypoints_with_background_img.csv', index=False)
