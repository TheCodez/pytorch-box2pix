import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision import models
from torchvision.models.googlenet import BasicConv2d, Inception

from models.multibox import MultiBox
from utils.helper import get_upsampling_weight


def box2pix(num_classes=11, pretrained=False):
    if pretrained:
        model = Box2Pix(num_classes)
        model.load_state_dict(model_zoo.load_url(''))
        return model

    return Box2Pix(num_classes)


class Box2Pix(nn.Module):
    """
        Implementation of Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes
            <https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18>
    """

    def __init__(self, num_classes=11):
        super(Box2Pix, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

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

        self.maxpool5 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception6a = Inception(1024, 256, 160, 320, 32, 128, 128)
        self.inception6b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.maxpool6 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception7a = Inception(1024, 256, 160, 320, 32, 128, 128)
        self.inception7b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.sem_score3b = nn.Conv2d(480, num_classes, kernel_size=1)
        self.sem_score4e = nn.Conv2d(832, num_classes, kernel_size=1)
        self.sem_score5b = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.sem_score6b = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.sem_upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.sem_upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.sem_upscore4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.sem_upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        self.offs_score3b = nn.Conv2d(480, 2, kernel_size=1)
        self.offs_score4e = nn.Conv2d(832, 2, kernel_size=1)
        self.offs_score5b = nn.Conv2d(1024, 2, kernel_size=1)
        self.offs_score6b = nn.Conv2d(1024, 2, kernel_size=1)
        self.offs_upscore = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False)
        self.offs_upscore2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False)
        self.offs_upscore4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False)
        self.offs_upscore8 = nn.ConvTranspose2d(2, 2, kernel_size=16, stride=8, bias=False)

        self.multibox = MultiBox(num_classes)
        self._initialize_weights(num_classes)

    def _initialize_weights(self, num_classes):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size[0] == 1 and m.out_channels in [num_classes, 2]:
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.ConvTranspose2d):
                upsampling_weight = get_upsampling_weight(m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(upsampling_weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_from_googlenet(self):
        googlenet = models.googlenet(pretrained=True)
        self.load_state_dict(googlenet.state_dict(), strict=False)

        for l1, l2 in zip([self.inception6b.modules(), self.inception7b.modules()],
                          [googlenet.inception5b.modules(), googlenet.inception5b.modules()]):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l1.weight.data.copy_(l2.weight.data)
                if l1.bias is not None:
                    l1.bias.data.copy_(l2.bias.data)

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
        inception3b = self.inception3b(x)
        x = self.maxpool3(inception3b)
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

        loc_preds, conf_preds = self.multibox(feature_maps)

        sem_score6b = self.sem_score6b(inception6b)
        sem_score5b = self.sem_score5b(inception5b)
        semantics = self.sem_upscore(sem_score6b)
        semantics = semantics[:, :, 1:1 + sem_score5b.size()[2], 1:1 + sem_score5b.size()[3]]
        semantics += sem_score5b
        sem_score4e = self.sem_score4e(inception4e)
        semantics = self.sem_upscore2(semantics)
        semantics = semantics[:, :, 1:1 + sem_score4e.size()[2], 1:1 + sem_score4e.size()[3]]
        semantics += sem_score4e
        sem_score3b = self.sem_score3b(inception3b)
        semantics = self.sem_upscore4(semantics)
        semantics = semantics[:, :, 1:1 + sem_score3b.size()[2], 1:1 + sem_score3b.size()[3]]
        semantics += sem_score3b
        semantics = self.sem_upscore8(semantics)
        semantics = semantics[:, :, 4:4 + size[2], 4:4 + size[3]].contiguous()

        offs_score6b = self.offs_score6b(inception6b)
        offs_score5b = self.offs_score5b(inception5b)
        offsets = self.offs_upscore(offs_score6b)
        offsets = offsets[:, :, 1:1 + offs_score5b.size()[2], 1:1 + offs_score5b.size()[3]]
        offsets += offs_score5b
        offs_score4e = self.offs_score4e(inception4e)
        offsets = self.offs_upscore2(offsets)
        offsets = offsets[:, :, 1:1 + offs_score4e.size()[2], 1:1 + offs_score4e.size()[3]]
        offsets += offs_score4e
        offs_score3b = self.offs_score3b(inception3b)
        offsets = self.offs_upscore4(offsets)
        offsets = offsets[:, :, 1:1 + offs_score3b.size()[2], 1:1 + offs_score3b.size()[3]]
        offsets += offs_score3b
        offsets = self.offs_upscore8(offsets)
        offsets = offsets[:, :, 4:4 + size[2], 4:4 + size[3]].contiguous()

        return loc_preds, conf_preds, semantics, offsets


if __name__ == '__main__':
    num_classes, width, height = 20, 1024, 2048

    model = Box2Pix(num_classes)  # .to('cuda')
    inp = torch.randn(1, 3, height, width)  # .to('cuda')

    loc, conf, sem, offs = model(inp)

    assert loc.size(2) == 4
    assert conf.size(2) == num_classes
    assert sem.size() == torch.Size([1, num_classes, height, width])
    assert offs.size() == torch.Size([1, 2, height, width])

    print('Pass size check.')
