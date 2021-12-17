import albumentations as A
import numpy as np
from PIL import Image
import cv2
from utils import plot_examples

if __name__ == '__main__':

    # In case using opencv for opening images you should convert from BGR to RGB
    # and no need to convert to numpy array (lines 34,35)

    img = Image.open("./images/elon.jpeg")
    mask = Image.open("./images/mask.jpeg")
    second_mask = Image.open("./images/second_mask.jpeg")

    transform = A.Compose([
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9),  # 40 degrees
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
    ])

    images_list = [img]
    # Should convert to numpy array:
    img = np.array(img)
    mask = np.array(mask)
    second_mask = np.array(second_mask)

    for i in range(3):
        # Adding here the mask,
        # therefore the transform will be applied on mask and actual image
        augmentations = transform(image=img, masks=[mask, second_mask])
        augmented_image = augmentations["image"]
        augmented_masks = augmentations["masks"]
        images_list.append(augmented_image)
        images_list.append(augmented_masks[0])
        images_list.append(augmented_masks[1])

    plot_examples(images_list)
