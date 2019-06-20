import torch.nn as nn

from loss.focalloss import FocalLoss


class BoxLoss(nn.Module):

    def __init__(self, num_classes=11, gamma=2, reduction='none'):
        super(BoxLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss(gamma, reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def forward(self, loc_pred, loc_target, conf_pred, labels_target):
        # find only non-background predictions
        positives = labels_target > 0
        predicted_loc = loc_pred[positives, :].reshape(-1, 4)
        groundtruth_loc = loc_target[positives, :].reshape(-1, 4)

        predicted_conf = conf_pred[positives, :].reshape(-1, self.num_classes)
        groundtruth_label = labels_target[positives, :]  # .reshape(-1, self.num_classes)

        loc_loss = self.l2_loss(predicted_loc, groundtruth_loc)
        conf_loss = self.focal_loss(predicted_conf, groundtruth_label)

        num_positives = loc_target.size(0)

        return loc_loss / num_positives, conf_loss / num_positives
