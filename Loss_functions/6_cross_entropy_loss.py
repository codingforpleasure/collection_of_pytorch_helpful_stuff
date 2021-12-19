import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from colorama import Fore, Style

def myCrossEntropyLoss(outputs, labels):
    batch_size = outputs.shape[0]
    outputs = F.log_softmax(outputs, dim=1)  # compute the log of softmax values
    outputs = outputs[range(batch_size), labels]  # pick the values corresponding to the labels
    return -torch.sum(outputs) / len(labels)


if __name__ == '__main__':
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)

    y_pred = np.array([[5.0, 4.0, 2.0],
                       [4.0, 2.0, 8.0],
                       [4.0, 4.0, 1.0]])
    input = torch.tensor(y_pred, requires_grad=True)

    target = np.array([0, 2, 1])
    target = torch.tensor(target)

    result = myCrossEntropyLoss(input, target)
    print(Fore.BLUE + "The result for my manually CrossEntropyLoss function is: " + Style.RESET_ALL,
          result.item())

    cross_entropy_loss = nn.CrossEntropyLoss()
    output = cross_entropy_loss(input, target)
    print(Fore.BLUE + "The result for my Pytorch CrossEntropyLoss is: " + Style.RESET_ALL,
          output.item())
