import albumentations as A
import numpy as np
from PIL import Image
import cv2
from utils import plot_examples

if __name__ == '__main__':

    img = Image.open("./images/elon.jpeg")

    transform = A.Compose([
        # You can add a manual transform of resizing ie transforms.Resize(), inside transforms.Compose() at the end.
        # This way, you ensure that all the images you stack will end up at the same size.
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
    ])

    images_list = [img]
    img = np.array(img)
    for i in range(15):
        augmentations = transform(image=img)
        augmented_image = augmentations["image"]
        images_list.append(augmented_image)

    plot_examples(images_list)
