import torch


def non_maximum_suppression(boxes, scores, threshold=0.5):

    if boxes.numel() == 0:
        return torch.LongTensor()

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    areas = (xmax - xmin) * (ymax - ymin)

    _, indices = scores.sort(0, descending=True)
    keep = []
    while indices.numel() > 0:
        i = indices[0]
        keep.append(i)

        if indices.numel() == 1:
            break

    return torch.LongTensor(keep)
