from __future__ import division

from typing import List

import numpy as np
import torch
import torch.nn as nn


class FCNHead(nn.Module):

    def __init__(self, num_channels):
        super(FCNHead, self).__init__()

        self.score3b = nn.Conv2d(480, num_channels, kernel_size=1)
        self.score4e = nn.Conv2d(832, num_channels, kernel_size=1)
        self.score5b = nn.Conv2d(1024, num_channels, kernel_size=1)
        self.score6b = nn.Conv2d(1024, num_channels, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=2, bias=False)
        self.upscore2 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=16, stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                upsampling_weight = get_upsampling_weight(m.out_channels, m.kernel_size[0])
                with torch.no_grad():
                    m.weight.copy_(upsampling_weight)

    def forward(self, size, outputs):
        # type: (List[int], List[torch.Tensor]) -> torch.Tensor

        score6b = self.score6b(outputs[3])
        score5b = self.score5b(outputs[2])
        output = self.upscore(score6b)
        output = output[:, :, 1:1 + score5b.shape[2], 1:1 + score5b.shape[3]]
        output += score5b
        score4e = self.score4e(outputs[1])
        output = self.upscore2(output)
        output = output[:, :, 1:1 + score4e.shape[2], 1:1 + score4e.shape[3]]
        output += score4e
        score3b = self.score3b(outputs[0])
        output = self.upscore4(output)
        output = output[:, :, 1:1 + score3b.shape[2], 1:1 + score3b.shape[3]]
        output += score3b
        output = self.upscore8(output)
        output = output[:, :, 4:4 + size[2], 4:4 + size[3]].contiguous()

        return output


def get_upsampling_weight(channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling

    Based on: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    weight = torch.zeros([channels, channels, kernel_size, kernel_size], dtype=torch.float64)
    weight[range(channels), range(channels), :, :] = filt

    return weight
