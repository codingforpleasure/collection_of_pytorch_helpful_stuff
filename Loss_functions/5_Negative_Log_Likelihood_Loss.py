# Reference:  https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


def my_softmax(vec):
    print(np.sum(np.exp(vec), axis=1) == (np.exp(2) + np.exp(4) + np.exp(8)))
    res = np.exp(vec) / np.sum(np.exp(vec), axis=1)
    return res


def my_NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum() / len(out)


if __name__ == '__main__':
    torch.set_printoptions(linewidth=120)
    np.set_printoptions(precision=5)
    torch.set_printoptions(precision=3)  # print floating point nicely
    np.set_printoptions(suppress=True)  # # prevent numpy exponential notation on print, default False

    # input is of size N x C = 3 x 3
    # C = Classes
    y_pred = np.array([[5.0, 4.0, 2.0],
                       [4.0, 2.0, 8.0],
                       [4.0, 4.0, 1.0]])
    y_pred_tensor = torch.tensor(y_pred, requires_grad=True)

    target = np.array([0, 2, 1])
    target = torch.tensor(target)
    # what the softmax does is that it squashes a vector of size K between 0 and 1
    # res = my_softmax(y_pred)
    # print("my_softmax result: \n", res)
    # print("Result after my_softmax and applying log: \n", np.log(res))

    log_softmax_result = F.log_softmax(y_pred_tensor, dim=1)
    print("Pytorch log_softmax result: \n", log_softmax_result)
    output = F.nll_loss(log_softmax_result, target)
    print("The output is: ", output)


    print("*" * 50)
    # each element in target has to have 0 <= value < C
    # Pay attention: the softmax function is used in tandem with the negative log-likelihood (NLL)


    y_pred = my_softmax(y_pred)
    res = my_NLLLoss(logs=y_pred, targets=torch.tensor(np.array([0, 2, 1])))

    y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
    log_softmax_output = F.log_softmax(y_pred_tensor)
    print("log_softmax_output: ", log_softmax_output)
    # each element in target has to have 0 <= value < C
    true_classes = torch.tensor([0, 2, 1])
    output = F.nll_loss(input=log_softmax_output,
                        target=true_classes)

    print("Negative Log Likelihood Loss: ", output)
