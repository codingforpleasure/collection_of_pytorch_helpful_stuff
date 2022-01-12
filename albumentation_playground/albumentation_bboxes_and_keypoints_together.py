import albumentations as A
import numpy as np
from PIL import Image
import cv2
from albumentations import BaseCompose

from utils2 import plot_examples

if __name__ == '__main__':
    img = cv2.imread("./images/twin_matches_ok.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Just for the purpose of the example I'm entering it manually:
    # you can insert list of bounding boxes such as: [[13, 170, 224, 410], [[130, 10, 24, 10]]]

    # bboxes = [[13, 170, 224, 410]]

    bboxes = [[
        45.150150150150154,
        24.393393393393396,
        234.03903903903904,
        48.11711711711712
    ],
        [
            23.82882882882882,
            35.504504504504496,
            45.150150150150154,
            227.99699699699698
        ]
    ]

    keypoints = np.array([
        (33.960861056751455, 39.138943248532286),
        (34.35225048923678, 224.46183953033267),
        (48.83365949119374, 36.007827788649706),
        (231.8082191780822, 36.007827788649706)
    ])

    # 1) `COCO` format
    #     `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
    #
    # 2) `Pascal_voc` format
    #     `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
    #
    # 3) `YOLO` format
    #     `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4]
    #
    # 4) The `albumentations` format
    #    is like `pascal_voc`, but normalized,
    #    in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].

    transform = A.Compose([
        A.Rotate(limit=40, p=0.9),  # 40 degress
    ]
        , bbox_params=A.BboxParams(format="pascal_voc",
                                   # minimum area of a bounding box. All bounding boxes whose
                                   # visible area in pixels is less than this value will be removed
                                   min_area=2048,
                                   # min_visibility: minimum fraction of area for a bounding box
                                   min_visibility=0.5,
                                   label_fields=[])  # <- Important for detection
        , keypoint_params=A.KeypointParams(format='xy'),
    )

    # Should convert to numpy array:
    img = np.array(img)
    images_list = [img]

    # result = isinstance(transform, BaseCompose)

    saved_bboxes = []
    bbox_all_images = [bboxes]
    keypoints_all_images = [keypoints]

    for i in range(2):
        augmentations = transform(image=img, bboxes=bboxes, keypoints=keypoints)
        augmented_image = augmentations["image"]
        bboxes_per_image = []

        if len(augmentations["bboxes"]):
            bbox_all_images.append(augmentations["bboxes"])

        # TODO: FIX IT make sure the all keypoints doesn't get out of the image!!

        if len(augmentations["keypoints"]):
            keypoints_all_images.append(np.array(augmentations["keypoints"]))

        images_list.append(augmented_image)

    # Lets write to csv file
    keypoints_all_images = np.array(keypoints_all_images)
    keypoints_all_images = keypoints_all_images.reshape(-1, 4)
    bbox_data = np.array(bbox_all_images)
    bbox_data = bbox_data.reshape(-1, 4)

    # concatenate two numpy arrays
    joined_bbox_and_keypoints = np.column_stack([keypoints_all_images, bbox_data])  # filenames,
    np.savetxt(
        fname="bla.txt",
        header='corner_up_x,corner_up_y,corner_down_x,corner_down_y,keypoint1-x,keypoint1-y,keypoint2-x,keypoint2-y',
        delimiter=',',
        X=joined_bbox_and_keypoints,
        fmt=["%f", "%f", "%f", "%f", "%f", "%f", "%f", "%f"],
        comments=""
    )

    plot_examples(images_list, bbox_all_images, keypoints_all_images)
