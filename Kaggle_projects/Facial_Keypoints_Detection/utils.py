import numpy as np
import pandas as pd
import torch
import config
from tqdm import tqdm


def get_submission(loader, dataset, model_15, model_4):
    model_15.eval()
    model_4.eval()
    id_lookup = pd.read_csv()

    for image, keypoints in tqdm(loader):
        image = image.to(config.DEVICE)
        preds_15 = torch.clip(model_15(image).squeeze(0), 0.0, 96.0)
        preds_4 = torch.clip(model_4(image).squeeze(0), 0.0, 96.0)


def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        num_examples += scores[]


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
