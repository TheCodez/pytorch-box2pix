import torch
import torchvision.datasets as datasets
import torchvision.transforms.functional as F

from utils.box_utils import get_bounding_box
from utils.box_coder import BoxCoder


class CityscapesDataset(datasets.Cityscapes):

    def __init__(self, root, split='train', joint_transform=None):
        super(CityscapesDataset, self).__init__(root, split, target_type=['instance', 'polygon'])

        self.joint_transform = joint_transform
        self.box_coder = BoxCoder()

    def __getitem__(self, index):
        image, target = super(CityscapesDataset, self).__getitem__(index)
        instance, json = target

        boxes, labels = self._create_boxes(json)
        boxes = self._normalize_boxes(image, boxes)

        if self.joint_transform:
            image, instance, boxes, labels = self.joint_transform(image, instance, boxes, labels)

        #instance = self.convert_id_to_train_id(instance)
        centroids = self._create_offsets(instance)
        boxes, labels = self.box_coder.encode(boxes, labels)

        return image, instance, centroids, boxes, labels

    @staticmethod
    def convert_id_to_train_id(target):
        target_copy = target.clone()

        for cls in CityscapesDataset.classes:
            target_copy[target == cls.id] = cls.train_id

        return target_copy

    @staticmethod
    def convert_train_id_to_id(target):
        target_copy = target.clone()

        for cls in CityscapesDataset.classes:
            target_copy[target == cls.train_id] = cls.id

        return target_copy

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

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def _create_offsets(self, inst):
        offsets = torch.zeros((2, inst.shape[0], inst.shape[1]))

        instance_ids = torch.unique(inst)
        instance_ids = instance_ids[instance_ids > 1000]

        for instance_id in instance_ids:
            mask = (inst == instance_id)
            instance_positions = torch.nonzero(mask, as_tuple=False).float().t()

            center_of_mass = instance_positions.mean(dim=1).reshape(2, 1)
            offsets[:, mask] = center_of_mass - instance_positions

        return offsets

    def _normalize_boxes(self, img, boxes):
        width, height = img.size
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height

        return boxes

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
                cmap[cls.id, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap
