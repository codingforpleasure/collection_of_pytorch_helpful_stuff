import numpy as np
import torch
from torch import nn


def BCE(y_pred, y_true):
    total_bce_loss = np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    # Getting the mean BCE loss
    num_of_samples = y_pred.shape[0]
    mean_bce_loss = total_bce_loss / num_of_samples
    return mean_bce_loss


if __name__ == '__main__':
    y_pred = np.array([0.1580, 0.4137, 0.2285])
    y_true = np.array([0.0, 1.0, 0.0])

    bce_value = BCE(y_pred, y_true)
    print("BCE error using Numpy is: " + str(bce_value))

    # How come do I need require grad, here ??
    y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
    y_true_tensor = torch.tensor(y_true)

    bce_loss = nn.BCELoss()
    output = bce_loss(input=y_pred_tensor, target=y_true_tensor)
    print("BCE error using Pytorch is: " + str(output.item()))
