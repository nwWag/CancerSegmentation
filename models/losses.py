import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    # Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        input = input.view(input.size(0), input.size(1), -
                           1)
        input = input.transpose(1, 2)
        input = input.contiguous().view(-1, input.size(2))

        target = target.view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryIOU(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        input = input.view(input.size(0), input.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1).squeeze(1)

        input = torch.argmax(F.log_softmax(input, dim=1), dim=1)

        inter = torch.sum((input == target) & (target == 1.0), dim=1)
        union = torch.sum(input, dim=1) + torch.sum(target, dim=1)
        iou = inter / union

        if self.size_average:
            return iou.mean()
        else:
            return iou.sum()
