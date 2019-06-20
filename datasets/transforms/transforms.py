import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

from datasets import CityscapesDataset


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, inst, boxes, labels):
        for t in self.transforms:
            img, inst, boxes, labels = t(img, inst, boxes, labels)
        return img, inst, boxes, labels


class ToTensor(object):
    def __call__(self, img, inst, boxes, labels):
        img = F.to_tensor(img)
        inst = torch.as_tensor(np.asarray(inst), dtype=torch.int64)

        return img, inst, boxes, labels


class ConvertIdToTrainId(object):

    def __call__(self, img, inst):
        inst = CityscapesDataset.convert_id_to_train_id(inst)

        return img, inst


class Resize(object):
    def __init__(self, new_size, old_size=(1024, 2048)):
        self.old_size = old_size
        self.new_size = new_size

        self.xscale = self.new_size[1] / self.old_size[1]
        self.yscale = self.new_size[0] / self.old_size[0]

    def __call__(self, img, inst, boxes, labels):
        img = F.resize(img, self.new_size, interpolation=Image.BILINEAR)
        inst = F.resize(inst, self.new_size, interpolation=Image.NEAREST)
        boxes = self._resize_boxes(boxes)

        return img, inst, boxes, labels

    def _resize_boxes(self, boxes):
        boxes = boxes.clone()
        boxes[:, 0] *= self.xscale
        boxes[:, 1] *= self.yscale
        boxes[:, 2] *= self.xscale
        boxes[:, 3] *= self.yscale

        return boxes


class Rescale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, inst, boxes, labels):
        width, height = img.size
        width *= self.scale
        height *= self.scale

        new_size = (int(height), int(width))
        img = F.resize(img, new_size, interpolation=Image.BILINEAR)
        inst = F.resize(inst, new_size, interpolation=Image.NEAREST)
        boxes = self._resize_boxes(boxes)

        return img, inst, boxes, labels

    def _resize_boxes(self, boxes):
        boxes = boxes.clone()
        boxes[:, 0] *= self.scale
        boxes[:, 1] *= self.scale
        boxes[:, 2] *= self.scale
        boxes[:, 3] *= self.scale

        return boxes


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, inst, boxes, labels):
        if random.random() < self.p:
            img = F.hflip(img)
            inst = F.hflip(inst)
            boxes = self._hflip_boxes(img.size[0], boxes)

        return img, inst, boxes, labels

    def _hflip_boxes(self, width, boxes):
        boxes = boxes.clone()
        box_width = boxes[:, 2] - boxes[:, 0]
        boxes[:, 2] = width - boxes[:, 0]
        boxes[:, 0] = boxes[:, 2] - box_width

        return boxes


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        img = self.transform(img)

        return img, target


class RandomGaussionBlur(object):
    def __init__(self, p=0.5, radius=0.8):
        self.p = p
        self.radius = radius

    def __call__(self, img, inst, boxes, labels):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))

        return img, inst, boxes, labels


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, inst, boxes, labels):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, inst, boxes, labels
