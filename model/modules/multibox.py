from typing import List, Tuple

import torch
import torch.nn as nn


class MultiBox(torch.jit.ScriptModule):

    __constants__ = ['num_classes', 'loc_layers', 'conf_layers']

    def __init__(self, num_classes):
        super(MultiBox, self).__init__()

        self.num_classes = num_classes
        num_defaults = [16, 16, 20, 21]
        in_channels = [832, 1024, 1024, 1024]

        loc = []
        conf = []
        for i in range(len(in_channels)):
            loc.append(nn.Conv2d(in_channels[i], num_defaults[i] * 4, kernel_size=1))
            conf.append(nn.Conv2d(in_channels[i], num_defaults[i] * num_classes, kernel_size=1))

        self.loc_layers = nn.ModuleList(loc)
        self.conf_layers = nn.ModuleList(conf)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    @torch.jit.script_method
    def forward(self, input):
        # type: (List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]

        locs = []
        confs = []

        #"""
        i = 0
        for loc_layer in self.loc_layers:
            loc = loc_layer(input[i])
            # (N x C x H x W) -> (N x H x W x C)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            locs.append(loc)
            i += 1

        i = 0
        for conf_layer in self.conf_layers:
            conf = conf_layer(input[i])
            # (N x C x H x W) -> (N x H x W x C)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1, self.num_classes)
            confs.append(conf)
            i += 1

        """
        i = 0
        for layer in input:
            loc = self.loc_layers[i](layer)
            # (N x C x H x W) -> (N x H x W x C)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.conf_layers[i](layer)
            # (N x C x H x W) -> (N x H x W x C)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1, self.num_classes)
            confs.append(conf)
            i += 1
        """

        loc_preds = torch.cat(locs, 1)
        conf_preds = torch.cat(confs, 1)

        return loc_preds, conf_preds


if __name__ == '__main__':
    box = MultiBox(11)
    print(box.code)
