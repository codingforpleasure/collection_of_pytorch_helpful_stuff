import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import albumentations as A

train_transforms = A.Compose([
    A.Resize(width=150, height=150),
    ToTensorV2(),
])


class Custom_dataset(Dataset):
    def __init__(self, data_dir, transform):
        filenames = os.listdir(data_dir)
        # store the full path to images
        self.full_path_to_filenames = [os.path.join(data_dir, file) for file in filenames]
        self.transform = transform

    def __len__(self):
        # Returns size of dataset
        return len(self.full_path_to_filenames)

    # Returns the transformed image at the given index and its corresponding label
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_path_to_filenames[idx])
        image = np.array(image)

        if self.transform:
            data = train_transforms(image=image)

        return data['image']


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data in tqdm(loader):
        data = data.type(torch.float64)
        data /= 255.0
        # Treating each channel in RGB separately
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_directory = "/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/diabetic-retinopathy-detection/all_images_resized_650"
    for image_type in ["valid", "train"]:
        print("***************", image_type, "***************")

        data_path = os.path.join(main_directory, f"{image_type}_650_size")
        train_set = Custom_dataset(data_path, train_transforms)

        # elem = next(iter(train_set))

        train_loader = DataLoader(dataset=train_set, batch_size=64)

        mean, std = get_mean_std(train_loader)
        print('mean: ', mean)
        print('std: ', std)
