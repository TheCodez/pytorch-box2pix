import torch
from ignite.metrics import Metric


class MeanAveragePrecision(Metric):

    def __init__(self, num_classes=20, output_transform=lambda x: x):
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform)

        self.num_classes = num_classes

    def reset(self):
        self._true_boxes = torch.tensor([], dtype=torch.long)
        self._true_labels = torch.tensor([], dtype=torch.long)

        self._det_boxes = torch.tensor([], dtype=torch.float32)
        self._det_labels = torch.tensor([], dtype=torch.float32)
        self._det_scores = torch.tensor([], dtype=torch.float32)

    def update(self, output):
        boxes_preds, labels_preds, scores_preds, boxes, labels = output

        self._true_boxes = torch.cat([self._true_boxes, boxes], dim=0)
        self._true_labels = torch.cat([self._true_labels, labels], dim=0)

        self._det_boxes = torch.cat([self._det_boxes, boxes_preds], dim=0)
        self._det_labels = torch.cat([self._det_labels, labels_preds], dim=0)
        self._det_scores = torch.cat([self._det_scores, scores_preds], dim=0)

    def compute(self):
        for c in range(1, self.num_classes):
            pass
