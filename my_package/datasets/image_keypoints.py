import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
import albumentations as A


class KeypointsDataset:
    def __init__(
            self,
            image_paths,
            keypoints,
            csv_file,
            classes=None,
            augmentations=None
    ):
        if isinstance(image_paths, list):
            self.image_paths = image_paths
        else:
            raise Exception("image_paths should be of type list")

        if csv_file is None and keypoints:
            if keypoints.ndim == 2 and keypoints.shape[1] == 2:  # x,y
                self.keypoints = keypoints
            else:
                raise Exception("keypoints data does not hold 2 columns")
        else:
            df = pd.read_csv(csv_file)
            if df.shape[1] == 2:
                self.keypoints = keypoints
            else:
                raise Exception("CSV file does not hold 2 columns")

        if augmentations is not None:
            if isinstance(augmentations, A.BaseCompose):
                self.augmentations = augmentations
            else:
                raise Exception("augmentations is not a valid Albumentation transform")

        self.classes = classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        img_path = self.image_paths[item]

        if not os.path.exists(img_path):
            raise Exception(img_path, " does not exist.")

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = self.keypoints[item]
        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image, keypoints=keypoints)
            image = augmented["image"]
            labels = augmented["keypoints"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        labels = np.array(labels)

        # if self.classes is None:
        #     labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        # else:
        #     labels = torch.tensor(self.classes[item], dtype=torch.int64)

        target = {"image": image, "labels": labels}

        target["image"] = torch.tensor(image, dtype=torch.float)
        return target
