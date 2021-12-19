# Well explained here:
# https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from colorama import Fore, Style


def my_logsoftmax(vec):
    # Softmax formula:
    # https://docs-assets.developer.apple.com/published/c2185dfdcf/0ab139bc-3ff6-49d2-8b36-dcc98ef31102.png
    softmax_result = np.exp(vec) / np.sum(np.exp(vec), axis=0)
    res = np.log(softmax_result)
    return res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def my_NLLLoss(logs, targets):
    # Retrieving the relevant values from the matrix logs in the correct class position
    out = logs[range(len(targets)), targets]
    return -out.sum() / len(out)


if __name__ == '__main__':
    # Here we have 5 classes (each column is a class)
    # input = torch.tensor([[1.6430, -1.1819, 0.8667, -0.5352, 0.2585],
    #                       [0.8617, -0.1880, -0.3865, 0.7368, -0.5482],
    #                       [-0.9189, -0.1265, 1.1291, 0.0155, -2.6702]], requires_grad=True)

    input = torch.tensor([[5, 4, 2],
                          [4, 2, 8],
                          [4, 4, 1]], requires_grad=True, dtype=torch.float64)

    target = torch.tensor([0, 2, 1])

    print("*" * 10, "Manually Calculation", "*" * 10)
    # sol = softmax(input.detach().numpy())
    sol = my_logsoftmax(input.detach().numpy())
    print(Fore.BLUE + "\n(1) The output of my manual function my_logsoftmax:\n " + Style.RESET_ALL, sol)
    result = my_NLLLoss(torch.tensor(sol), targets=target)
    print(Fore.BLUE + "\n(2) The output of Pytorch function myNLLLoss: " + Style.RESET_ALL, result.item(), "\n")
    # every element in target should have 0 <= value < C

    print("*" * 10, "Pytorch Calculation", "*" * 10)

    m = nn.LogSoftmax(dim=0)
    print(Fore.BLUE + "\n(1) The output of Pytorch function nn.LogSoftmax:\n " + Style.RESET_ALL, m(input))

    nll_loss = nn.NLLLoss()
    output = nll_loss(m(input), target)
    print(Fore.BLUE + "\n(2) The output of Pytorch function nn.NLLLoss: " + Style.RESET_ALL, output.item())
