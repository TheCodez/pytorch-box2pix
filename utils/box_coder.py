import torch

from utils import box_utils


class BoxCoder(object):
    """
        References:
            https://github.com/ECer23/ssd.pytorch
            https://github.com/kuangliu/torchcv
    """

    def __init__(self, img_size=(1024, 2048)):
        self.variances = (0.1, 0.2)

        self.img_height, self.img_width = img_size
        self.feature_maps = [(128, 64), (64, 32), (32, 16), (16, 8)]
        self.priors = self.get_prior_boxes()

    def get_prior_boxes(self):
        priors = [
            # height, width
            (4, 52), (24, 24), (54, 8), (80, 22), (52, 52),
            (20, 78), (156, 50), (78, 78), (48, 144), (412, 76),
            (104, 150), (74, 404), (644, 166), (358, 448), (70, 686), (68, 948),
            (772, 526), (476, 820), (150, 1122), (890, 880), (516, 1130)
        ]

        boxes = []
        for width, height in self.feature_maps:
            step_w = width / self.img_width
            step_h = height / self.img_height
            for x in range(width):
                for y in range(height):
                    for p_h, p_w in priors:
                        cx = (x + 0.5) * step_w
                        cy = (y + 0.5) * step_h
                        h = p_h / self.img_height
                        w = p_w / self.img_width

                        boxes.append((cx, cy, h, w))

        return torch.as_tensor(boxes)

    def encode(self, boxes, labels, change_threshold=0.7):
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

        loc_xy = (boxes[:, :2] - priors[:, :2]) / priors[:, 2:] / self.variances[0]
        loc_wh = torch.log(boxes[:, 2:] / priors[:, 2:]) / self.variances[1]
        loc = torch.cat([loc_xy, loc_wh], 1)

        conf = labels[max_idx] + 1  # background class = 0
        conf[change < change_threshold] = 0  # background

        return loc, conf

    def decode(self, loc_preds, conf_preds, score_thresh=0.6, nms_thresh=0.5):
        """Decode predicted loc/cls back to real box locations and class labels.
            Args:
              loc_preds: (tensor) predicted loc, sized [8732,4].
              conf_preds: (tensor) predicted conf, sized [8732,21].
              score_thresh: (float) threshold for object confidence score.
              nms_thresh: (float) threshold for box nms.

            Returns:
                boxes: (tensor) bbox locations, sized [#obj,4].
                labels: (tensor) class labels, sized [#obj,].
        """
        cxcy = loc_preds[:, :2] * self.variances[0] * self.priors[:, 2:] + self.priors[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * self.variances[1]) * self.priors[:, 2:]
        box_preds = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = conf_preds.size(1)
        for i in range(num_classes - 1):
            score = conf_preds[:, i + 1]  # class i corresponds to (i + 1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box  # torchvision.layers.nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.full(box[keep].size()[0], i, dtype=torch.int64))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)

        return boxes, labels, scores


if __name__ == '__main__':
    coder = BoxCoder()
    print(coder.priors[4:])
    print(coder.priors.size())
