import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for Dense Object Detection
    <https://arxiv.org/abs/1708.02002>

    .. math::
        \text{loss}(p_{t}) = -(1-p_{t})^ \gamma \cdot \log(p_{t})

    Args:
        gamma (int, optional): Gamma smoothly adjusts the rate at which easy examples
            are down weighted. If gamma is equals 0 it's the same as cross entropy loss. Default: 1
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    """

    def __init__(self, gamma=1, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        log_pt = -F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(log_pt)

        loss = -torch.pow(1 - pt, self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
