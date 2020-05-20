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


def my_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

def __d_change(prior, ground_truth):
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


def d_change(priors, gt):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf

    Input should be in point form (xmin, ymin, xmax, ymax).
    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for
    """
    num_priors = priors.size(0)
    num_gt = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat = gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -torch.sqrt((diff ** 2).sum(dim=2))

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
