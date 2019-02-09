import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

        self.uncert_semantics = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.uncert_offsets = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.uncert_ssdbox = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.uncert_ssdclass = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, semantics_loss, offsets_loss, box_loss, conf_loss):
        loss1 = 0.5 * torch.exp(-self.uncert_semantics) * semantics_loss + self.uncert_semantics
        loss2 = torch.exp(-self.uncert_offsets) * offsets_loss + self.uncert_offsets
        loss3 = torch.exp(-self.uncert_ssdbox) * box_loss + self.uncert_ssdbox
        loss4 = 0.5 * torch.exp(-self.uncert_ssdclass) * conf_loss + self.uncert_ssdclass

        loss = loss1 + loss2 + loss3 + loss4

        return loss
