import math
import os

import numpy as np  # many operations are more concise in matrix form
import PIL
from PIL import Image, ImageDraw


def get_rotation_matrix(angle):
    """ For background, https://en.wikipedia.org/wiki/Rotation_matrix
    rotation is clockwise in traditional descartes, and counterclockwise,
    if y goes down (as in picture coordinates)
    """
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])


def my_rotate(points, pivot, angle):
    """ Get coordinates of points rotated by a given angle counterclocwise

    Args:
        points (np.array): point coordinates shaped (n, 2)
        pivot (np.array): [x, y] coordinates of rotation center
        angle (float): counterclockwise rotation angle in radians
    Returns:
        np.array of new coordinates shaped (n, 2)
    """
    relative_points = points - pivot
    return relative_points.dot(get_rotation_matrix(angle)) + pivot


if __name__ == '__main__':

    background = Image.open(
        '/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/backgrounds/background2.jpg')
    object_image = Image.open("/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/matches/match1.png")
    my_angle_deg = 30
    my_angle = math.radians(my_angle_deg)

    rsize_x, rsize_y = object_image.size
    # to get shift introduced by rotation+clipping we'll need to rotate all four corners
    # starting from top-right corners, counter-clockwise
    collection_of_keypoints = np.array([
        [rsize_x, 0],  # top-right
        [0, 0],  # top-left
        [0, rsize_y],  # bottom-left
        [rsize_x, rsize_y]  # bottom-right
    ])
    # rectangle_corners now are:
    # array([[262,   0],
    #       [  0,   0],
    #       [  0,  67],
    #       [262,  67]])

    rotated_corners = my_rotate(collection_of_keypoints, collection_of_keypoints[0], my_angle)
    # as a result of rotation, one of the corners might end up left from 0,
    # e.g. if the rectangle is really tall and rotated 90 degrees right
    # or, leftmost corner is no longer at 0, so part of the canvas is clipped
    shift_introduced_by_rotation_clip = rotated_corners.min(axis=0)

    rotated_shifted_corners = rotated_corners - shift_introduced_by_rotation_clip

    # finally, demo
    # this part is just copied from the question
    rectangle_rotate = Image.Image.rotate(object_image, angle=my_angle_deg, resample=Image.BICUBIC, expand=True)

    # box: 2-tuple giving the upper left corner
    px = int(background.size[0] / 2)
    py = int(background.size[1] / 2)
    background.paste(im=rectangle_rotate,
                     box=(px, py),
                     mask=rectangle_rotate)

    # let's see if dots land right after these translations:
    draw_img_pts = ImageDraw.Draw(background)
    r = 10
    for point in rotated_shifted_corners:
        pos_xNew, pos_yNew = point + [px, py]
        draw_img_pts.ellipse((pos_xNew - r, pos_yNew - r, pos_xNew + r, pos_yNew + r), fill='red')

    background.save('example_with_roatation.png')
