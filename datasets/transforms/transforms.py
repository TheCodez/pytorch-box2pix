import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage.filters import gaussian


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
        inst = F.to_tensor(inst).long()

        return img, inst, boxes, labels


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


class RandomGaussionBlur(object):
    def __init__(self, sigma=(0.15, 1.15)):
        self.sigma = sigma

    def __call__(self, img, inst, boxes, labels):
        sigma = self.sigma[0] + random.random() * self.sigma[1]
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        img = Image.fromarray(blurred_img.astype(np.uint8))

        return img, inst, boxes, labels


class RandomScale(object):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, img, inst, boxes, labels):
        scale = random.uniform(1.0, self.scale)

        img = F.affine(img, 0, (0, 0), scale, 0)
        inst = F.affine(inst, 0, (0, 0), scale, 0)

        return img, inst, boxes, labels


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        from torchvision import transforms
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, inst, boxes, labels):
        img = self.transform(img)

        return img, inst, boxes, labels
