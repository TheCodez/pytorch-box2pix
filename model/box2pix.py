import torch
from torch.utils import model_zoo

from modules.fcn import FCNHead
from modules.googlenet import GoogLeNet
from modules.multibox import MultiBox


def box2pix(num_classes=11, pretrained=False, init_googlenet=False):
    if pretrained:
        model = Box2Pix(num_classes, init_googlenet=False)
        model.load_state_dict(model_zoo.load_url(''))
        return model
    elif init_googlenet:
        model = Box2Pix(num_classes, init_googlenet=True)
        return model
    return Box2Pix(num_classes)


class Box2Pix(torch.jit.ScriptModule):
    """
        Implementation of Box2Pix: Single-Shot Instance Segmentation by Assigning Pixels to Object Boxes
            <https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18>
    """

    def __init__(self, num_classes=11, init_googlenet=False):
        super(Box2Pix, self).__init__()

        self.googlenet = GoogLeNet(init_googlenet)
        self.offsets = FCNHead(2)
        self.semantics = FCNHead(num_classes)
        self.multibox = MultiBox(num_classes)

    @torch.jit.script_method
    def forward(self, x):
        size = x.size()

        feature_maps, outputs = self.googlenet(x)
        loc_preds, conf_preds = self.multibox(feature_maps)
        semantics = self.semantics(size, outputs)
        offsets = self.offsets(size, outputs)

        return loc_preds, conf_preds, semantics, offsets


if __name__ == '__main__':
    num_classes, width, height = 20, 1024, 2048

    model = Box2Pix(num_classes, init_googlenet=True)  # .to('cuda')
    inp = torch.randn(1, 3, height, width)  # .to('cuda')

    loc, conf, sem, offs = model(inp)

    assert loc.size(2) == 4
    assert conf.size(2) == num_classes
    assert sem.size() == torch.Size([1, num_classes, height, width])
    assert offs.size() == torch.Size([1, 2, height, width])

    print('Pass size check.')
