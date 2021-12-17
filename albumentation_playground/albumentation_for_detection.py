import albumentations as A
import numpy as np
from PIL import Image
import cv2
from utils import plot_examples

if __name__ == '__main__':
    img = Image.open("./images/cat.jpg")

    # Just for the purpose of the example I'm entering it manually:
    # you can insert list of bounding boxes such as: [[13, 170, 224, 410], [[130, 10, 24, 10]]]
    bboxes = [[13, 170, 224, 410]]

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
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9),  # 40 degress
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25,
                   g_shift_limit=25,
                   b_shift_limit=25,
                   p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                # minimum area of a bounding box. All bounding boxes whose
                                # visible area in pixels is less than this value will be removed
                                min_area=2048,
                                # min_visibility: minimum fraction of area for a bounding box
                                min_visibility=0.5,
                                label_fields=[])  # <- Important for detection
    )

    # Should convert to numpy array:
    img = np.array(img)
    images_list = [img]

    saved_bboxes = [bboxes[0]]
    for i in range(10):
        augmentations = transform(image=img, bboxes=bboxes)
        augmented_image = augmentations["image"]
        if len(augmentations["bboxes"]):
            # The [0] is just because I'm lazy, we have here always a single bbox in the image
            saved_bboxes.append(augmentations["bboxes"][0])
            images_list.append(augmented_image)

    plot_examples(images_list, saved_bboxes)
