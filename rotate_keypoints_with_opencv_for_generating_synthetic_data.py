# Some of the logic was taken from here:
# https://towardsdatascience.com/geometric-transformations-in-computer-vision-an-intuitive-explanation-with-python-examples-b0b6f06e1844
# IT works well!

import glob
import cv2
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    for filename_full_path in glob.glob(
            "/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/matches/*.png"):
        img = cv2.imread(filename_full_path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(filename_full_path)

        center = (img.shape[1] // 2, img.shape[0] // 2)  # Get the image center
        rotation_matrix = cv2.getRotationMatrix2D(center, -45, 1)  # Calculate the rotation matrix
        new_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        original_four_corners = [(0, 0, 1), (img.shape[1], 0, 1), (0, img.shape[0], 1), (img.shape[1], img.shape[0], 1)]
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

        new_img2 = cv2.warpAffine(img, rotation_matrix, new_dimensions)

        cv2.imwrite("/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/example.png", new_img2)

        original_keypoints_on_objects = pd.read_csv('./matches/matches_keypoints.csv', index_col='filename')

        coordinates_object = dict(original_keypoints_on_objects.loc[filename,])

        collection_of_original_keypoints = np.array([
            [coordinates_object['keypoint1-x'], coordinates_object['keypoint1-y'], 1],
            [coordinates_object['keypoint2-x'], coordinates_object['keypoint2-y'], 1],
        ])

        new_keypoints = [np.matmul(rotation_matrix, pt) for pt in collection_of_original_keypoints]

        for rotated_point in new_keypoints:
            # Convert to homogenous coordinates in np array format first so that you can pre-multiply M
            cv2.circle(new_img2, (int(rotated_point[0]), int(rotated_point[1])), 30, (0, 0, 0, 255), -1)

        cv2.imwrite(filename[:-4] + '_keypoint_rotated.png', new_img2)
