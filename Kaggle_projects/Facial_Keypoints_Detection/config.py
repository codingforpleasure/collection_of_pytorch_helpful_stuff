import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 4
# We build two sepreately models we are using efficienet b0,
# Here for 4 keypoints
CHECKPOINT_FILE = "b0_4.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True

train_transforms = A.Compose(
    [
        # the orignal size
        A.Resize(width=96, height=96),
        # # important: BORDER_CONSTANT otherwise will mess the keypoints!!!
        # A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        # A.IAAAffine(shear=15, scale=1.0, mode="constant", p=0.2),
        # A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        # A.OneOf([
        #     A.GaussNoise(p=0.8),
        #     A.CLAHE(p=0.8),
        # ], p=1.0),
        # # Efficient b0 receieves 3 image channels, therefore we are repeating the channels
        # so efficientnet will accept it and creae 3 channels per image
        A.Normalize(
            mean=[0.4899, 0.4899, 0.4899],
            std=[0.2327, 0.2327, 0.2327],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
)

valid_transforms = A.Compose(
    [
        # the orignal size
        A.Resize(width=96, height=96),
        A.Normalize(
            mean=[0.4899, 0.4899, 0.4899],
            std=[0.2327, 0.2327, 0.2327],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
)