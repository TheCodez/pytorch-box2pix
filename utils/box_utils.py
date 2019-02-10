import torch

priors = [
    # height, width
    (4, 52),
    (25, 25),
    (54, 8),
    (80, 22),
    (52, 52),
    (20, 78),
    (156, 50),
    (78, 78),
    (48, 144),
    (412, 76),
    (104, 150),
    (74, 404),
    (645, 166),
    (358, 448),
    (70, 686),
    (68, 948),
    (772, 526),
    (476, 820),
    (150, 1122),
    (890, 880),
    (518, 1130)
]

priors_new = [
    # height, width
    (4, 52),
    (24, 24),
    (54, 8),
    (80, 22),
    (52, 52),
    (20, 78),
    (156, 50),
    (78, 78),
    (48, 144),
    (412, 76),
    (104, 150),
    (74, 404),
    (644, 166),
    (358, 448),
    (70, 686),
    (68, 948),
    (772, 526),
    (476, 820),
    (150, 1122),
    (890, 880),
    (516, 1130)
]


def get_bounding_box(polygon):
    fpoint = polygon[0]
    xmin, ymin, xmax, ymax = fpoint[0], fpoint[1], fpoint[0], fpoint[1]
    for point in polygon:
        x, y = point[0], point[1]
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x)
        ymax = max(ymax, y)

    return xmin, ymin, xmax, ymax


def d_change(prior, ground_truth):
    """Compute a change based metric of two sets of boxes.

    Args:
        prior (tensor): Prior boxes, Shape: [num_priors, 4]
        ground_truth (tensor): Ground truth bounding boxes, Shape: [num_objects, 4]
    """

    xtl = torch.abs(prior[:, 0] - ground_truth[:, 0])
    ytl = torch.abs(prior[:, 1] - ground_truth[:, 1])
    xbr = torch.abs(prior[:, 2] - ground_truth[:, 2])
    ybr = torch.abs(prior[:, 3] - ground_truth[:, 3])

    wgt = ground_truth[:, 2] - ground_truth[:, 0]
    hgt = ground_truth[:, 3] - ground_truth[:, 1]

    return torch.sqrt((torch.pow(ytl, 2) / hgt) + (torch.pow(xtl, 2) / wgt)
                      + (torch.pow(ybr, 2) / hgt) + (torch.pow(xbr, 2) / wgt))


def corner_to_center_form(boxes):
    """Convert bounding boxes from (xmin, ymin, xmax, ymax) to (cx, cy, width, height)

    Args:
        boxes (tensor): Boxes, Shape: [num_priors, 4]
    """

    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,
                      boxes[:, 2:] - boxes[:, :2]], 1)


def center_to_corner_form(boxes):
    """Convert bounding boxes from (cx, cy, width, height) to (xmin, ymin, xmax, ymax)

    Args:
        boxes (tensor): Boxes, Shape: [num_priors, 4]
    """

    return torch.cat([boxes[:, 2:] - (boxes[:, :2] / 2),
                      boxes[:, 2:] + (boxes[:, :2] / 2)], 1)


def nms(boxes, scores, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, indices = scores.scores.sort(0, descending=True)

    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)

        xx1 = torch.max(x1[i], x1[indices[1:]])
        yy1 = torch.max(y1[i], y1[indices[1:]])
        xx2 = torch.min(x2[i], x2[indices[1:]])
        yy2 = torch.min(y2[i], y2[indices[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[indices[1:]] - inter)

        inds = torch.nonzero(ovr <= thresh).squeeze()
        indices = indices[inds + 1]

    return keep


if __name__ == '__main__':
    """
    layers = [
        # inception4e
        {'size': 427, 'boxes': []},
        # inception5b
        {'size': 715, 'boxes': []},
        # inception6b
        {'size': 1291, 'boxes': []},
        # inception7b
        {'size': 2443, 'boxes': []}
    ]

    # calculate the number of associated prior boxes for each layer
    for prior in priors_new:
        height, width = prior

        for layer in layers:
            if layer['size'] > (height * 2) or layer['size'] > (width * 2):
                layer['boxes'].append(prior)

    for layer in layers:
        print('Number of priors: ', len(layer['boxes']))
    """
