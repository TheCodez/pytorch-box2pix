from __future__ import print_function

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import os
import os.path
import errno

import json
import os

import torch.utils.data as data
from PIL import Image
from matplotlib.patches import Rectangle

from datasets.transforms.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor


class KITTI(data.Dataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        train (bool, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, train=True, target_type='instance', joint_transform=None, img_transform=None):
        self.root = os.path.expanduser(root)
        split = 'training' if train else 'testing'
        self.images_dir = os.path.join(self.root, split, 'image_2')
        self.targets_dir = os.path.join(self.root, split, target_type)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_type = target_type
        self.train = train
        self.images = []
        self.targets = []

        if target_type not in ['instance', 'semantic', 'semantic_rgb']:
            raise ValueError('Invalid value for "target_type"! Valid values are: "instance", "semantic"'
                             ' or "semantic_rgb"')

        if not os.path.isdir(self.images_dir) and not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            if train:
                self.targets.append(os.path.join(self.targets_dir, file_name))

    def __getitem__(self, index):
        if self.train:
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.targets[index])

            boxes = [(325, 170, 475, 240), (555, 165, 695, 220), (720, 155, 900, 240)]
            confs = torch.tensor([1, 1, 1])

            if self.joint_transform:
                image, target, boxes = self.joint_transform(image, target, boxes)

            if self.img_transform:
                image = self.img_transform(image)

            return image, target, boxes, confs
        else:
            image = Image.open(self.images[index]).convert('RGB')

            if self.img_transform:
                image = self.img_transform(image)

            return image

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        #ToTensor()
    ])

    dataset = KITTI('../data/kitti', train=True, joint_transform=joint_transforms)
    img, inst, bboxes, confs = dataset[10]

    #print('Box size: ', bboxes.size())
    #print('Instance size: ', inst.size())
    #img = torchvision.transforms.functional.to_pil_image(img)
    #plt.imshow(img)

    #inst = torchvision.transforms.functional.to_pil_image(inst)
    plt.imshow(inst)
    ax = plt.gca()

    for i, box in enumerate(bboxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
