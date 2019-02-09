import torch
import torch.nn as nn


class MultiBox(nn.Module):

    def __init__(self, num_classes):
        super(MultiBox, self).__init__()

        self.num_classes = num_classes
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        num_defaults = [16, 16, 20, 21]
        in_channels = [832, 1024, 1024, 1024]

        for i in range(len(in_channels)):
            self.loc_layers.append(nn.Conv2d(in_channels[i], num_defaults[i] * 4, kernel_size=1))
            self.conf_layers.append(nn.Conv2d(in_channels[i], num_defaults[i] * num_classes, kernel_size=1))

    def forward(self, input):
        loc_preds = []
        conf_preds = []

        for i, layer in enumerate(input):
            loc = self.loc_layers[i](layer)
            # (N x C x H x W) -> (N x H x W x C)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            loc_preds.append(loc)

            conf = self.conf_layers[i](layer)
            # (N x C x H x W) -> (N x H x W x C)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1, self.num_classes)
            conf_preds.append(conf)

        loc_preds = torch.cat(loc_preds, 1)
        conf_preds = torch.cat(conf_preds, 1)

        return loc_preds, conf_preds
