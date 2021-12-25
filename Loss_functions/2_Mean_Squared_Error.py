# Reference: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

import numpy as np
import torch
from torch import nn


# Defining Mean Square Error loss function
def mse(pred, true):
    # Find absolute difference
    differences = pred - true
    absolute_differences_sqr = np.absolute(differences ** 2)
    # find the absolute mean
    mean_absolute_error = absolute_differences_sqr.mean()
    return mean_absolute_error


def sse(pred, true):
    # Find absolute difference
    differences = pred - true
    absolute_differences_sqr = np.absolute(differences ** 2)
    return np.sum(absolute_differences_sqr)


if __name__ == '__main__':
    y_pred = np.array([0.000, 0.100, 0.200])
    y_true = np.array([0.000, 0.200, 0.250])

    mse_value = mse(y_pred, y_true)
    print("MSE error using Numpy is: " + str(mse_value))

    # How come do I need require grad, here ??
    y_pred_tensor = torch.tensor(y_pred, requires_grad=False)
    y_true_tensor = torch.tensor(y_true)

    mse_loss = nn.MSELoss()  # reduction=mean (default value)
    output = mse_loss(input=y_pred_tensor, target=y_true_tensor)
    print("MSE error using Pytorch is: " + str(output.item()))

    print("*" * 50)

    sse_value = sse(y_pred, y_true)
    print("SSE error using Numpy is: " + str(sse_value))

    mse_loss = nn.MSELoss(reduction="sum")  # reduction=sum (default value)
    output = mse_loss(input=y_pred_tensor, target=y_true_tensor)
    print("SSE (Sum Square Error) error using Pytorch is: " + str(output.item()))
