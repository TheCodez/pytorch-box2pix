import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from models.multibox import MultiBox
from utils.helper import get_upsampling_weight


def box2pix(num_classes=20, pretrained=False):
    if pretrained:
        model = Box2Pix(num_classes)
        model.load_state_dict(model_zoo.load_url(''))
        return model

    return Box2Pix(num_classes)


class Box2Pix(nn.Module):
    """
        Implementation of
        Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes
            <https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18>
    """

    def __init__(self, num_classes=20):
        super(Box2Pix, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.maxpool5 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # padding=1)
        self.inception6a = Inception(1024, 256, 160, 320, 64, 128, 128)
        self.inception6b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.maxpool6 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # padding=1)
        self.inception7a = Inception(1024, 256, 160, 320, 32, 128, 128)
        self.inception7b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.offs_score7b = nn.Conv2d(1024, 2, kernel_size=1)
        self.offs_upscore2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False)
        self.offs_score6b = nn.Conv2d(1024, 2, kernel_size=1)
        self.offs_upscore16 = nn.ConvTranspose2d(2, 2, kernel_size=32, stride=16, bias=False)

        self.sem_score7b = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.sem_upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.sem_score6b = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.sem_upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)

        self.multibox = MultiBox(num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.ConvTranspose2d):
                upsampling_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(upsampling_weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def copy_weights(self, googlenet):
        for name, m in googlenet.named_modules():
            if hasattr(self, name):
                setattr(self, name, m)

    def forward(self, x):
        feature_maps = []
        size = x.size()

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        inception4e = self.inception4e(x)
        feature_maps.append(inception4e)

        x = self.maxpool4(inception4e)
        x = self.inception5a(x)
        inception5b = self.inception5b(x)
        feature_maps.append(inception5b)

        x = self.maxpool5(inception5b)
        x = self.inception6a(x)
        inception6b = self.inception6b(x)
        feature_maps.append(inception6b)

        x = self.maxpool6(inception6b)
        x = self.inception7a(x)
        inception7b = self.inception7b(x)
        feature_maps.append(inception7b)

        for f in feature_maps:
            print(f.size())

        loc_preds, conf_preds = self.multibox(feature_maps)

        # TODO: use learnable upsampling
        sem_score7b = self.sem_score7b(inception7b)
        sem_score6b = self.sem_score6b(inception6b)
        score = F.interpolate(sem_score7b, sem_score6b.size()[2:], mode='bilinear', align_corners=True)
        score += sem_score6b
        semantic = F.interpolate(score, size[2:], mode='bilinear', align_corners=True)

        offs_score7b = self.offs_score7b(inception7b)
        offs_score6b = self.offs_score6b(inception6b)
        score = F.interpolate(offs_score7b, offs_score6b.size()[2:], mode='bilinear', align_corners=True)
        score += offs_score6b
        offsets = F.interpolate(score, size[2:], mode='bilinear', align_corners=True)

        """
        sem_score7b = self.sem_score7b(inception7b)
        sem_upscore2 = self.sem_upscore2(sem_score7b)
        sem_score6b = self.sem_score6b(inception6b)
        sem_score6b_crop = sem_score6b[:, :, 5:5 + sem_upscore2.size()[2], 5:5 + sem_upscore2.size()[3]]
        sem_upscore2 += sem_score6b_crop
        semantic = self.sem_upscore16(sem_upscore2)
        semantic = semantic[:, :, 27:27 + size[2], 27:27 + size[3]].contiguous()

        offs_score7b = self.offs_score7b(inception7b)
        offs_upscore2 = self.offs_upscore2(offs_score7b)
        offs_score6b = self.offs_score6b(inception6b)
        offs_score6b_crop = offs_score6b[:, :, 5:5 + offs_upscore2.size()[2], 5:5 + offs_upscore2.size()[3]]
        offs_upscore2 += offs_score6b_crop
        offsets = self.sem_upscore16(offs_upscore2)
        offsets = offsets[:, :, 27:27 + size[2], 27:27 + size[3]].contiguous()
        """

        return loc_preds, conf_preds, semantic, offsets


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1, stride=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    num_classes, width, height = 20, 512, 1024

    model = Box2Pix(num_classes)#.to('cuda')
    inp = torch.randn(1, 3, height, width)#.to('cuda')

    loc, conf, sem, offs = model(inp)

    assert loc.size(2) == 4
    assert conf.size(2) == num_classes
    assert sem.size() == torch.Size([1, num_classes, height, width])
    assert offs.size() == torch.Size([1, 2, height, width])

    print('Pass size check.')
