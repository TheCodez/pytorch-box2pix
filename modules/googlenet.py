from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GoogLeNet(torch.jit.ScriptModule):

    def __init__(self, transform_input=False, init_from_googlenet=False):
        super(GoogLeNet, self).__init__()
        self.transform_input = torch.jit.Attribute(transform_input, bool)

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.maxpool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception6a = Inception2(1024, 256, 160, 320, 32, 128, 128)
        self.inception6b = Inception2(832, 384, 192, 384, 48, 128, 128)

        self.maxpool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception7a = Inception2(1024, 256, 160, 320, 32, 128, 128)
        self.inception7b = Inception2(832, 384, 192, 384, 48, 128, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if init_from_googlenet:
            self._init_from_googlenet()

    def _init_from_googlenet(self):
        googlenet = models.googlenet(pretrained=True)
        self.load_state_dict(googlenet.state_dict(), strict=False)
        self.transform_input = True

    @torch.jit.script_method
    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        return torch.cat([x_ch0, x_ch1, x_ch2], 1)

    @torch.jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tuple[List[Tensor], Dict[str, Tensor]]

        feature_maps = []
        outputs = torch.jit.annotate(Dict[str, torch.Tensor], {})

        if self.transform_input:
            x = self._transform_input(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        outputs['inception3b'] = self.inception3b(x)
        x = self.maxpool3(outputs['inception3b'])
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        outputs['inception4e'] = self.inception4e(x)
        feature_maps.append(outputs['inception4e'])

        x = self.maxpool4(outputs['inception4e'])
        x = self.inception5a(x)
        outputs['inception5b'] = self.inception5b(x)
        feature_maps.append(outputs['inception5b'])

        x = self.maxpool5(outputs['inception5b'])
        x = self.inception6a(x)
        outputs['inception6b'] = self.inception6b(x)
        feature_maps.append(outputs['inception6b'])

        x = self.maxpool6(outputs['inception6b'])
        x = self.inception7a(x)
        inception7b = self.inception7b(x)
        feature_maps.append(inception7b)

        return feature_maps, outputs


class Inception(torch.jit.ScriptModule):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    @torch.jit.script_method
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(torch.jit.ScriptModule):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    @torch.jit.script_method
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception2(torch.jit.ScriptModule):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception2, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
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

    @torch.jit.script_method
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
