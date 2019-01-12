import torch.nn as nn
import torch.nn.functional as F

from loss.focalloss import FocalLoss


class MultiBoxLoss(nn.Module):

    def __init__(self):
        super(MultiBoxLoss, self).__init__()

        self.focal_loss = FocalLoss(gamma=2)

    def forward(self, loc_pred, conf_pred, loc_target, conf_target):

        # find only non-background predictions
        positives = conf_target > 0
        predicted_loc = loc_pred[positives, :].reshape(-1, 4)
        groundtruth_loc = loc_target[positives, :].reshape(-1, 4)

        loc_loss = F.mse_loss(predicted_loc, groundtruth_loc, size_average=False)
        conf_loss = self.focal_loss(conf_pred, conf_target)

        return loc_loss, conf_loss
