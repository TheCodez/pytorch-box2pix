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


def hard_negative_mining(conf_loss, positives):
    batch_size = positives.size(0)

    conf_loss[positives] = 0
    conf_loss = conf_loss.view(batch_size, -1)

    _, indices = conf_loss.sort(1, descending=True)


if __name__ == '__main__':

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
