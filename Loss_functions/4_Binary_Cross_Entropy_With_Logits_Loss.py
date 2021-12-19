import numpy as np
import torch
from torch import nn


def BCE_and_logit_loss(y_pred, y_true):
    bce_loss = nn.BCELoss()
    # Applying to the input Sigmoid function
    output = bce_loss(input=torch.sigmoid(y_pred), target=y_true)
    return output


if __name__ == '__main__':
    y_pred = np.array([0.1580, 0.4137, 0.2285])
    y_true = np.array([0.0, 1.0, 0.0])  # target

    # How come do I need require grad, here ??
    y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
    y_true_tensor = torch.tensor(y_true)

    output = BCE_and_logit_loss(y_pred_tensor, y_true_tensor)
    print("BCE with Logit loss error applying sigmoid on BCELoss with Pytorch is: " + str(output.item()))

    bcelog_loss = nn.BCEWithLogitsLoss()
    output = bcelog_loss(input=y_pred_tensor, target=y_true_tensor)

    print("BCE with Logit loss error using Pytorch is: " + str(output.item()))
