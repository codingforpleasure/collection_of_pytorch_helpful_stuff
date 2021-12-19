import numpy as np
# from torch.nn import L1Loss
import torch
from torch import nn


# Defining Mean Absolute Error loss function
def mae(pred, true):
    # Find absolute difference
    differences = pred - true
    absolute_differences = np.absolute(differences)
    # find the absolute mean
    mean_absolute_error = absolute_differences.mean()
    return mean_absolute_error


if __name__ == '__main__':
    y_pred = np.array([0.000, 0.100, 0.200])
    y_true = np.array([0.000, 0.200, 0.250])

    mae_value = mae(y_pred, y_true)
    print("MAE error using Numpy is: " + str(mae_value))

    # How come do I need require grad, here ??
    y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
    y_true_tensor = torch.tensor(y_true)
    loss = nn.L1Loss()
    output = loss(y_pred_tensor, y_true_tensor)

    print("MAE error using Pytorch is: " + str(output.item()))
