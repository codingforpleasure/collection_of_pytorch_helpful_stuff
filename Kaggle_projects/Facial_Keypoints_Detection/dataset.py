import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import config
import matplotlib.pyplot as plt
import cv2

KEYPOINT_COLOR = (0, 255, 0)  # Green


class FacialKeypointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        idx_columns_relevant = self.data.isnull().sum() == 0
        self.category_names = self.data.columns[idx_columns_relevant][:-1]
        self.transform = transform
        self.train = train
        self.num_keypoints = int(os.path.basename(csv_file).split('_')[1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        filename = self.data.loc[item, "filename"]
        image = Image.open(f"./training/keypoints_{self.num_keypoints}/{filename}")
        image = np.array(image)
        # Since the image should be 3 channels when we pass it into the model EfficientNet,
        # we will repeat the image:

        image = np.repeat(image.reshape(96, 96, 1), repeats=3, axis=2)

        keypoints = self.data.loc[item, self.category_names]
        keypoints = np.array(keypoints).reshape(-1, 2)
        keypoints = list(zip(keypoints[:, 0], keypoints[:, 1]))

        if self.transform:
            augmentations = self.transform(image=image, keypoints=keypoints)
            image = augmentations["image"]
            keypoints = augmentations["keypoints"]

        # Important: because we will compare those keypoints to the predictions of the model
        # therefore we now will reshape it to 1 dimensional array:

        # Moreover since the network predicts points of type np.float32 we will
        # make sure our keypoints are of the same type

        keypoints = np.array(keypoints, dtype=np.float32).reshape(-1)

        return {"image": image,
                "labels": keypoints}


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=1):
    # keypoints should be of the following format

    image = image.copy()

    # keypoints = zip(keypoints[:, 0], keypoints[:, 1])
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    csv_file = "train_15_keypoints.csv"
    number_of_keypoints = int(csv_file[6:].split('_')[0])
    # train_4 = pd.read_csv(csv_file)
    #
    # # Either "train_15.csv" (With 15 keypoints)
    # # or "train_4.csv" (With 4 keypoints)
    # # csv_file = "train_15_keypoints.csv"
    # train_15 = pd.read_csv(csv_file)
    # print(train_15.columns)

    my_dataset = FacialKeypointDataset(csv_file=csv_file,
                                       train=True,
                                       transform=config.train_transforms)

    print(next(iter(my_dataset)))
    loader = DataLoader(my_dataset, batch_size=15, num_workers=0, shuffle=True)
    batch = next(iter(loader))

    imgs_per_batch, keypoints_per_batch = batch['image'], batch['labels']
    print("imgs_per_batch.shape: ", imgs_per_batch.shape)
    print("keypoints_per_batch.shape: ", keypoints_per_batch.shape)

    # for img, keypoints_per_image in zip(batch['image'], batch['labels']):
    #     single_img = np.array(img, dtype=np.int8).squeeze()
    #     print("single_img.shape: ", single_img.shape)
    #     print("keypoints_per_image.shape: ", keypoints_per_image.shape)
    #     coordinnates_keypoints = np.array(batch['labels'])
    #     vis_keypoints(single_img, keypoints_per_image)
