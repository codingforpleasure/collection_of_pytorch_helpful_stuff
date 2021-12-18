import cv2
import torch
import numpy as np
from PIL import Image


class ImageDataset:
    """
    :param image_paths: list of paths to images
    :param targets: numpy array
    :param augmentations: albumentations augmentations
    """

    def __init__(
            self,
            image_paths,
            targets,
            augmentations=None,
            backend="pil",
            is_channel_first=True,
            is_grayscale=False,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend
        self.is_channel_first = is_channel_first
        self.is_grayscale = is_grayscale

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, item):
            targets = self.targets[item]

            if self.backend == "pil":
                image = Image.open(self.image_paths[item])
                image = np.array(image)
                if self.augmentations is not None:
                    augmented = self.augmentations(image=image)
                    image = augmented["image"]
            elif self.backend == "cv2":
                if self.grayscale is False:
                    image = cv2.imread(self.image_paths[item])
                    # important:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imread(self.image_paths[item], cv2.IMREAD_GRAYSCALE)

                if self.augmentations is not None:
                    augmented = self.augmentations(image=image)
                    image = augmented["image"]
            else:
                raise Exception("Backend not implemented")

            if self.is_channel_first is True and self.is_grayscale is False:
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)

            image_tensor = torch.tensor(image)

            if self.is_grayscale:
                image_tensor = image_tensor.unsqueeze(0)
            return {
                "image": image_tensor,
                "targets": torch.tensor(targets),
            }
