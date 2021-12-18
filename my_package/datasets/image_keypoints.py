import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile


class KeypointsDataset:
    def __init__(
            self,
            image_paths,
            keypoints,
            csv_file,
            classes=None,
            augmentations=None
    ):
        self.image_paths = image_paths

        if csv_file is None and keypoints:
            self.keypoints = keypoints
        else:
            self.keypoints = pd.read_csv(csv_file)
            
        self.augmentations = augmentations
        self.classes = classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        image = cv2.imread(self.image_paths[item])
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
