import numpy as np
from ignite.metrics import Metric


class IntersectionOverUnion(Metric):
    """Computes the intersection over union (IoU) per class.

        based on: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

        - `update` must receive output of the form `(y_pred, y)`.
    """

    def __init__(self, num_classes=10, ignore_index=255, output_transform=lambda x: x):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

        super(IntersectionOverUnion, self).__init__(output_transform=output_transform)

    def _fast_hist(self, label_true, label_pred):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = label_true != self.ignore_index
        hist = np.bincount(self.num_classes * label_true[mask].astype(np.int) + label_pred[mask],
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, output):
        y_pred, y = output

        for label_true, label_pred in zip(y.numpy(), y_pred.numpy()):
            self.confusion_matrix += self._fast_hist(label_true.flatten(), label_pred.flatten())

    def compute(self):
        hist = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        return np.nanmean(iu)
