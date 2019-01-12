import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from matplotlib.patches import Rectangle

from datasets.transforms.transforms import Resize, RandomHorizontalFlip, Compose, ToTensor
from utils.box_utils import get_bounding_box


class CityscapesClass(object):
    def __init__(self, name, id, train_id, category, cat_id, has_instances, ignore_in_eval, color):
        self.name = name
        self.id = id
        self.train_id = train_id
        self.category = category
        self.cat_id = cat_id
        self.has_instances = has_instances
        self.ignore_in_eval = ignore_in_eval
        self.color = color


class CityscapesDataset(datasets.Cityscapes):
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        #CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142))
    ]

    def __init__(self, root, split='train', joint_transform=None, img_transform=None):
        self.joint_transform = joint_transform
        self.img_transform = img_transform

        super(CityscapesDataset, self).__init__(root, split, target_type=['instance', 'polygon'])

    def __getitem__(self, index):
        image, target = super(CityscapesDataset, self).__getitem__(index)
        instance, json = target

        instance = self._convert_id_to_train_id(instance)
        bboxes, confs = self._create_boxes(json)

        if self.joint_transform:
            image, instance, boxes = self.joint_transform(image, instance, bboxes)

        if self.img_transform:
            image = self.img_transform(image)

        return image, instance, bboxes, confs

    def _convert_id_to_train_id(self, instance):
        instance = np.array(instance)
        instance_copy = instance.copy()
        for cls in self.classes:
            instance_copy[instance == cls.id] = cls.train_id
        instance = Image.fromarray(instance_copy.astype(np.uint8))

        return instance

    def _create_boxes(self, json):
        boxes = []
        confs = []
        objects = json['objects']
        for obj in objects:
            polygons = obj['polygon']
            cls = self.get_class_from_name(obj['label'])
            if cls and cls.has_instances:
                boxes.append(get_bounding_box(polygons))
                confs.append(cls.id)

        return torch.as_tensor(boxes, dtype=torch.float32), torch.tensor(confs)

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
        return [cls.name for cls in CityscapesDataset.classes if cls.has_instances]


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        Resize((512, 1024)),
        #RandomHorizontalFlip(),
        ToTensor()
    ])

    dataset = CityscapesDataset('../data/cityscapes', split='train', joint_transform=joint_transforms)
    # 2700 is Ulm University
    img, inst, bboxes, confs = dataset[2699]

    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    print('Box size: ', bboxes.size())
    print('Instance size: ', inst.size())

    #img = Lighting(0.1, eigval, eigvec)(img)
    #img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = torchvision.transforms.functional.to_pil_image(img)
    #img = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4)(img)
    plt.imshow(img)
    ax = plt.gca()

    """
    for i, box in enumerate(bboxes):
        cls = dataset.get_class_from_id(confs[i])
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 1, cls.name,
                # bbox=dict(facecolor='r', alpha=0.2),
                fontsize=8, color='red')
    """

    priors = [
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


    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(21)
    for i, box in enumerate(priors):
        height, width = box

        rect = Rectangle((200, 50), width * 0.5, height * 0.5, linewidth=1, edgecolor=cmap(i), facecolor='none')
        ax.add_patch(rect)

    plt.show()
