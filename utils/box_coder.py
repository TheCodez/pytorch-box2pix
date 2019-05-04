from typing import Tuple, List

import torch

from utils import box_utils


@torch.jit.script
def nms(bboxes, scores, threshold=0.5):
    # type: (Tensor, Tensor, float) -> Tensor
    """Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
    Returns:
      keep: (tensor) selected indices.
    Ref:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = torch.jit.annotate(List[int], [])
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        # if order.numel() == 1:
        #    break

        xx1 = x1[order[1:]]  # .clamp(min=x1[i])
        yy1 = y1[order[1:]]  # .clamp(min=y1[i])
        xx2 = x2[order[1:]]  # .clamp(max=x2[i])
        yy2 = y2[order[1:]]  # .clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (ovr <= threshold).nonzero().squeeze()
        # if ids.numel() == 0:
        #    break
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.int64)


@torch.jit.script
class FeatureMapDef(object):
    def __init__(self, width, height, receptive_size):
        # type: (int, int, int) -> None
        self.width = width
        self.height = height
        self.receptive_size = receptive_size


@torch.jit.script
class BoxCoder(object):
    """
        References:
            https://github.com/ECer23/ssd.pytorch
            https://github.com/kuangliu/torchcv
    """

    def __init__(self, img_width=2048, img_height=1024):
        # type: (int, int) -> None
        self.variances = (0.1, 0.2)

        priors = [
            # height, width
            (4, 52), (24, 24), (54, 8), (80, 22), (52, 52),
            (20, 78), (156, 50), (78, 78), (48, 144), (412, 76),
            (104, 150), (74, 404), (644, 166), (358, 448), (70, 686), (68, 948),
            (772, 526), (476, 820), (150, 1122), (890, 880), (516, 1130)
        ]

        feature_maps = [
            FeatureMapDef(128, 64, 427),
            FeatureMapDef(64, 32, 715),
            FeatureMapDef(32, 16, 1291),
            FeatureMapDef(16, 8, 2443)
        ]

        boxes = torch.jit.annotate(List[Tuple[float, float, float, float]], [])
        for fm in feature_maps:
            step_w = fm.width / img_width
            step_h = fm.height / img_height
            for x in range(fm.width):
                for y in range(fm.height):
                    for p in priors:
                        p_h, p_w = p
                        cx = (x + 0.5) * step_w
                        cy = (y + 0.5) * step_h
                        h = p_h / img_height
                        w = p_w / img_width

                        if fm.receptive_size > (p_h * 2) or fm.receptive_size > (p_w * 2):
                            boxes.append((cx, cy, h, w))

        self.priors = torch.randn(4324, 4)  # torch.as_tensor(boxes, dtype=torch.float32).clamp_(0.0, 1.0)

    def encode(self, boxes, labels, change_threshold=0.7):
        # type: (Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
        """Encode target bounding boxes and class labels.
            SSD coding rules:
                tx = (x - anchor_x) / (variance[0] * anchor_w)
                ty = (y - anchor_y) / (variance[0] * anchor_h)
                tw = log(w / anchor_w) / variance[1]
                th = log(h / anchor_h) / variance[1]

            Args:
                boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [#obj, 4].
                labels: (tensor) object class labels, sized [#obj,].
                change_threshold: (float) the change metric threshold
       """

        priors = self.priors
        priors = box_utils.center_to_corner_form(priors)

        change = box_utils.d_change(boxes, priors)

        change, max_idx = change.max(0)
        max_idx.squeeze_(0)
        change.squeeze_(0)

        boxes = boxes[max_idx]
        boxes = box_utils.corner_to_center_form(boxes)
        priors = box_utils.corner_to_center_form(priors)

        loc_xy = (boxes[:, :2] - priors[:, :2]) / (self.variances[0] * priors[:, 2:])
        loc_wh = torch.log(boxes[:, 2:] / priors[:, 2:]) / self.variances[1]
        loc = torch.cat([loc_xy, loc_wh], 1)

        conf = labels[max_idx] + 1  # background class = 0
        conf[change < change_threshold] = torch.zeros(1)  # 0  # background

        return loc, conf

    def decode(self, loc_preds, conf_preds, score_thresh=0.6, nms_thresh=0.5):
        # type: (Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor]

        """Decode predicted loc/cls back to real box locations and class labels.
            Args:
              loc_preds: (tensor) predicted loc, sized [8732,4].
              conf_preds: (tensor) predicted conf, sized [8732,21].
              score_thresh: (float) threshold for object confidence score.
              nms_thresh: (float) threshold for box nms.

            Returns:
                boxes_pred: (tensor) bbox locations, sized [#obj,4].
                labels_pred: (tensor) class labels, sized [#obj,].
                scores_pred: (tensor) label scores, sized [#obj,].
        """
        cxcy = loc_preds[:, :2] * self.variances[0] * self.priors[:, 2:] + self.priors[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * self.variances[1]) * self.priors[:, 2:]
        box_preds = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)

        boxes = [torch.zeros(1)]
        labels = [torch.zeros(1)]
        scores = [torch.zeros(1)]
        num_classes = conf_preds.size(1)
        for i in range(num_classes - 1):
            score = conf_preds[:, i + 1]  # class i corresponds to (i + 1) column
            mask = score > score_thresh
            # if not mask.any():
            #    continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = nms(box, score, nms_thresh)  # torchvision.ops.nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.full([box[keep].size(0)], i, dtype=torch.int64))
            scores.append(score[keep])

        boxes_pred = torch.cat(boxes, 0)
        labels_pred = torch.cat(labels, 0)
        scores_pred = torch.cat(scores, 0)

        return boxes_pred, labels_pred, scores_pred


if __name__ == '__main__':
    coder = BoxCoder()
    print(coder.priors[4:])
    print(coder.priors.size())
