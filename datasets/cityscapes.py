import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image

from utils.box_utils import get_bounding_box


class CityscapesDataset(datasets.Cityscapes):

    def __init__(self, root, split='train', joint_transform=None, img_transform=None):
        super(CityscapesDataset, self).__init__(root, split, target_type=['instance', 'polygon'])

        self.joint_transform = joint_transform
        self.img_transform = img_transform

    def __getitem__(self, index):
        image, target = super(CityscapesDataset, self).__getitem__(index)
        instance, json = target

        instance = self._convert_id_to_train_id(instance)
        boxes, labels = self._create_boxes(json)

        if self.joint_transform:
            image, instance, boxes, labels = self.joint_transform(image, instance, boxes, labels)

        if self.img_transform:
            image = self.img_transform(image)

        return image, instance, boxes, labels

    def _convert_id_to_train_id(self, instance):
        instance = np.array(instance)
        instance_copy = instance.copy()

        for cls in self.classes:
            instance_copy[instance == cls.id] = cls.train_id
        instance = Image.fromarray(instance_copy.astype(np.uint8))

        return instance

    def _create_boxes(self, json):
        boxes = []
        labels = []
        objects = json['objects']
        for obj in objects:
            polygons = obj['polygon']
            cls = self.get_class_from_name(obj['label'])
            if cls and cls.has_instances:
                boxes.append(get_bounding_box(polygons))
                labels.append(cls.id)

        return torch.as_tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    @staticmethod
    def get_class_from_name(name):
        for cls in CityscapesDataset.classes:
            if cls.name == name:
                return cls
        return None

    @staticmethod
    def get_class_from_id(id):
        for cls in CityscapesDataset.classes:
            if cls.id == id:
                return cls
        return None

    @staticmethod
    def get_instance_classes():
        return [cls for cls in CityscapesDataset.classes if cls.has_instances]

    @staticmethod
    def num_instance_classes():
        return len(CityscapesDataset.get_instance_classes())

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in CityscapesDataset.classes:
            if cls.has_instances:
                cmap[cls.trainId, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap
