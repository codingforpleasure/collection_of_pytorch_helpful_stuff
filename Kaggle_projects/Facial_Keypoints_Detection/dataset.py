import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config


class FacialKeypointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.category_names = self.data.columns
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        image = Image(f"training/face_{item}.jpeg")
        keypoints = self.data.iloc[item, :-1]
        if self.transform:
            augmentations = self.transform(image=image, keypoints=keypoints)
            image = augmentations["image"]
            labels = augmentations["keypoints"]


if __name__ == '__main__':
    # Either "train_15.csv" (With 15 keypoints)
    # or "train_4.csv" (With 4 keypoints)
    my_dataset = FacialKeypointDataset(csv_file="train_15.csv", train=True, transform=config.train_transforms)
    loader = DataLoader(my_dataset, batch_size=64, num_workers=0, shuffle=True)
    print(next(iter(loader)))
