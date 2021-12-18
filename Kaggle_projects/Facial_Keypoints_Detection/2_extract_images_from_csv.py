import numpy as np
import pandas as pd
from PIL import Image


def get_images_in_training_set():
    df = pd.read_csv('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/facial-keypoints-detection/training/training.csv')

    print("Column in dataset is: ", df.columns)
    # The input image is given in the last field of the data files,
    # and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels.
    images = df['Image']

    for row_idx in range(df.shape[0]):
        img = df.loc[row_idx, 'Image'].split()

        # convert to int
        img = list(map(int, img))
        img_array = np.array(img, dtype=np.uint8).reshape(96, 96)
        im = Image.fromarray(img_array)
        im.save(f"training/face_{row_idx}.jpeg")


def get_images_in_testing_set():
    df = pd.read_csv('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/facial-keypoints-detection/test/test.csv')

    print("Column in dataset is: ", df.columns)
    # The input image is given in the last field of the data files,
    # and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels.
    images = df['Image']

    for row_idx in range(df.shape[0]):
        img = df.loc[row_idx, 'Image'].split()

        # convert to int
        img = list(map(int, img))
        img_array = np.array(img, dtype=np.uint8).reshape(96, 96)
        im = Image.fromarray(img_array)
        im.save(f"testing/face_{row_idx}.jpeg")


if __name__ == '__main__':
    get_images_in_training_set()
    get_images_in_testing_set()
