import torch


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
        prior (tensor): Prior boxes, Shape: [num_boxes, 4]
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
        boxes (tensor): Boxes, Shape: [num_boxes, 4]
    """

    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,
                      boxes[:, 2:] - boxes[:, :2]], dim=1)


def center_to_corner_form(boxes):
    """Convert bounding boxes from (cx, cy, width, height) to (xmin, ymin, xmax, ymax)

    Args:
        boxes (tensor): Boxes, Shape: [num_boxes, 4]
    """

    return torch.cat([boxes[:, 2:] - (boxes[:, :2] / 2),
                      boxes[:, 2:] + (boxes[:, :2] / 2)], dim=1)


def resize_boxes(boxes, image):
    """Resize normalized boxes back to the original image size

    Args:
        boxes (tensor): Boxes, Shape: [num_boxes, 4]
        image: (tensor): Image, Shape: [batch_size, 3, height, width]
    """
    height, width = image.size()[2:]
    return torch.cat([boxes[:, 0] * width, boxes[:, 1] * height,
                      boxes[:, 2] * width, boxes[:, 3] * height], dim=1)
