import torch
from ignite.metrics import Metric


class MeanAveragePrecision(Metric):

    def __init__(self, num_classes=20, iou_threshold=0.5, output_transform=lambda x: x):
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform)

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

    def reset(self):
        self._true_boxes = torch.tensor([], dtype=torch.float32)
        self._true_labels = torch.tensor([], dtype=torch.long)

        self._det_boxes = torch.tensor([], dtype=torch.float32)
        self._det_labels = torch.tensor([], dtype=torch.long)
        self._det_scores = torch.tensor([], dtype=torch.float32)

    def update(self, output):
        boxes_preds, labels_preds, scores_preds, boxes, labels = output

        self._true_boxes = torch.cat([self._true_boxes, boxes], dim=0)
        self._true_labels = torch.cat([self._true_labels, labels], dim=0)

        self._det_boxes = torch.cat([self._det_boxes, boxes_preds], dim=0)
        self._det_labels = torch.cat([self._det_labels, labels_preds], dim=0)
        self._det_scores = torch.cat([self._det_scores, scores_preds], dim=0)

    def compute(self):
        # from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
        true_images = []
        for i in range(self._true_labels.size(0)):
            true_images.extend([i] * self._true_labels[i].size(0))
        true_images = torch.LongTensor(true_images)

        det_images = []
        for i in range(self._det_labels.size(0)):
            det_images.extend([i] * self._det_labels[i].size(0))
        det_images = torch.LongTensor(det_images)

        average_precisions = torch.zeros((self.num_classes - 1), dtype=torch.float)
        for c in range(1, self.num_classes):
            true_class_images = true_images[self._true_labels == c]
            true_class_boxes = self._true_boxes[self._true_labels == c]

            true_class_boxes_detected = torch.zeros(true_class_boxes.size(0), dtype=torch.uint8)

            det_class_images = det_images[self._det_labels == c]
            det_class_boxes = self._det_boxes[self._det_labels == c]
            det_class_scores = self._det_scores[self._det_labels == c]
            n_class_detections = det_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
            det_class_images = det_class_images[sort_ind]
            det_class_boxes = det_class_boxes[sort_ind]

            true_positives = torch.zeros(n_class_detections, dtype=torch.float)
            false_positives = torch.zeros(n_class_detections, dtype=torch.float)
            for d in range(n_class_detections):
                this_detection_box = det_class_boxes[d].unsqueeze(0)
                this_image = det_class_images[d]

                object_boxes = true_class_boxes[true_class_images == this_image]
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                overlaps = self._box_iou(this_detection_box, object_boxes)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

                if max_overlap.item() > self.iou_threshold:
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[
                            original_ind] = 1
                    else:
                        false_positives[d] = 1
                else:
                    false_positives[d] = 1

            cumul_true_positives = torch.cumsum(true_positives, dim=0)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10)
            cumul_recall = cumul_true_positives / self.num_classes

            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c - 1] = precisions.mean()

        mean_average_precision = average_precisions.mean().item()

        return mean_average_precision

    def _box_iou(self, boxes1, boxes2):
        # based on torchvision

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou
