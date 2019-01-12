import random

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, inst, boxes):
        for t in self.transforms:
            img, inst, boxes = t(img, inst, boxes)
        return img, inst, boxes


class ToTensor(object):

    def __call__(self, img, inst, boxes):
        img = F.to_tensor(img)
        inst = F.to_tensor(inst).long()
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)

        return img, inst, boxes


class Resize(object):

    def __init__(self, new_size):
        self.old_size = (1024, 2048)
        self.new_size = new_size

        self.xscale = self.new_size[1] / self.old_size[1]
        self.yscale = self.new_size[0] / self.old_size[0]

    def __call__(self, img, inst, boxes):
        img = F.resize(img, self.new_size, interpolation=Image.BILINEAR)
        inst = F.resize(inst, self.new_size, interpolation=Image.NEAREST)
        boxes = self._resize_boxes(boxes)

        return img, inst, boxes

    def _resize_boxes(self, boxes):
        boxes[:, 0] *= self.xscale
        boxes[:, 1] *= self.yscale
        boxes[:, 2] *= self.xscale
        boxes[:, 3] *= self.yscale

        return boxes


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, inst, boxes):
        if random.random() < self.p:
            img = F.hflip(img)
            inst = F.hflip(inst)
            boxes = self._hflip_boxes(img.size[0], boxes)

        return img, inst, boxes

    def _hflip_boxes(self, width, boxes):
        box_width = boxes[:, 2] - boxes[:, 0]
        boxes[:, 2] = width - boxes[:, 0]
        boxes[:, 0] = boxes[:, 2] - box_width

        return boxes
