import glob
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from utils import load_checkpoint, save_checkpoint, get_submission

import numpy as np
from tqdm import tqdm
import config
import dataset
from efficientnet_pytorch import EfficientNet  # https://github.com/lukemelas/EfficientNet-PyTorch
from torch.utils.data import DataLoader

from Detecting_Facial_Keypoints.utils import get_rmse


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.to(targets.DEVICE)

        # forward
        preds = model(loader)

        loss = loss_fn(preds, targets)
        ##???
        losses.append(loss.item())

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f" Loss average over epoch: gh")


if __name__ == '__main__':
    for csv_file in glob.glob("*_keypoints.csv"):
        print("csv_file: ", csv_file)
        train_ds = dataset.FacialKeypointDataset(csv_file=csv_file,
                                                 train=True,
                                                 transform=config.train_transforms)

        train_loader = DataLoader(dataset=train_ds,
                                  shuffle=True,
                                  num_workers=config.NUM_WORKERS,
                                  pin_memory=config.PIN_MEMORY,
                                  batch_size=config.BATCH_SIZE)

        val_ds = dataset.FacialKeypointDataset(csv_file=csv_file,
                                               train=True,
                                               transform=config.train_transforms)

        val_loader = DataLoader(dataset=train_ds,
                                shuffle=False,
                                num_workers=config.NUM_WORKERS,
                                pin_memory=config.PIN_MEMORY,
                                batch_size=config.BATCH_SIZE)

    my_loss_fn = nn.MSELoss(reduction='sum')
    model = EfficientNet.from_pretrained("efficientnet-b0")

    for param in model.parameters():
        param.requires_grad = False

    model._fc = nn.Linear(in_features=1280, out_features=30)

    model.to(config.DEVICE)

    # For binary classification tasks, SGD and Adam optimizers are used the most
    # The cnn_model.parameters() returns an iterator over module parameters that are passed to the optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        get_rmse(loader=val_loader,
                 model=model,
                 device=config.DEVICE,
                 loss_fn=my_loss_fn)

        train_one_epoch(loader=train_loader,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        loss_fn=my_loss_fn,
                        device=config.DEVICE)

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(state=checkpoint,
                            filename=config.CHECKPOINT_FILE)
