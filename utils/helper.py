import numpy as np
import torch


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


@torch.jit.script
def assign_pix2box(semantics, offsets, boxes, labels):
    return semantics
