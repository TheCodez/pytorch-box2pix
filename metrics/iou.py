import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class ConfusionMatrix(Metric):
    def __init__(self, num_classes, output_transform=lambda x: x):
        self._num_examples = 0
        self.num_classes = num_classes
        self.confusion_matrix = None
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def update(self, output):
        y_pred, y = output

        y_pred = y_pred.argmax(1).flatten()
        y = y.flatten()

        n = self.num_classes
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros((n, n), dtype=torch.int64, device=y.device)

        with torch.no_grad():
            k = (y >= 0) & (y < n)
            inds = n * y[k].to(torch.int64) + y_pred[k]
            self.confusion_matrix += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            self._num_examples += y_pred.shape[0]

    def reset(self):
        if self.confusion_matrix is not None:
            self.confusion_matrix.zero_()
        self._num_examples = 0

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        return self.confusion_matrix.cpu()


def IoU(cm):
    # Increase floating point precision
    cm = cm.type(torch.float64)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    return iou


def mIoU(cm):
    return IoU(cm=cm).mean()
