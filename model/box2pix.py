import torch
import torch.hub as hub
import torch.nn as nn

from model.modules import GoogLeNet, FCNHead, MultiBox, Inception2


def box2pix(num_classes=11, pretrained=False, init_googlenet=False):
    model = Box2Pix(num_classes, init_googlenet)
    if pretrained:
        model.load_state_dict(hub.load_state_dict_from_url(''))
    return model


class Box2Pix(nn.Module):
    """
        Implementation of Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes
            <https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18>
    """

    def __init__(self, num_classes=11, init_googlenet=False):
        super(Box2Pix, self).__init__()

        self.googlenet = GoogLeNet(init_googlenet)

        self.maxpool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception6a = Inception2(1024, 256, 160, 320, 32, 128, 128)
        self.inception6b = Inception2(832, 384, 192, 384, 48, 128, 128)

        self.maxpool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception7a = Inception2(1024, 256, 160, 320, 32, 128, 128)
        self.inception7b = Inception2(832, 384, 192, 384, 48, 128, 128)

        self.offsets = FCNHead(2)
        self.semantics = FCNHead(num_classes)
        self.multibox = MultiBox(num_classes)

    def forward(self, x):
        size = x.size()

        feature_maps = self.googlenet(x)

        x = self.maxpool5(feature_maps[-1])
        x = self.inception6a(x)
        inception6b = self.inception6b(x)
        feature_maps.append(inception6b)

        x = self.maxpool6(inception6b)
        x = self.inception7a(x)
        inception7b = self.inception7b(x)
        feature_maps.append(inception7b)

        loc_preds, conf_preds = self.multibox(feature_maps)
        semantics = self.semantics(size, feature_maps)
        offsets = self.offsets(size, feature_maps)

        return loc_preds, conf_preds, semantics, offsets


if __name__ == '__main__':
    num_classes, width, height = 20, 2048, 1024

    model = Box2Pix(num_classes, init_googlenet=True)  # .to('cuda')
    inp = torch.randn(1, 3, height, width)  # .to('cuda')

    loc, conf, sem, offs = model(inp)

    assert loc.size(2) == 4
    assert conf.size(2) == num_classes
    assert sem.size() == torch.Size([1, num_classes, height, width])
    assert offs.size() == torch.Size([1, 2, height, width])

    print('Pass size check.')
