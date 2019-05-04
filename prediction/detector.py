import torch
import torch.nn.functional as F

from models.box2pix import box2pix
from utils import helper
from utils.box_coder import BoxCoder


class Detector(torch.jit.ScriptModule):
    __constants__ = ['debug']

    def __init__(self, score_thresh=0.6, nms_thresh=0.5):
        super(Detector, self).__init__()
        # self.box_coder = BoxCoder(2048, 1024)
        self.net = box2pix(pretrained=False)
        self.net.eval()

        self.score_thresh = torch.jit.Attribute(score_thresh, float)
        self.nms_thresh = torch.jit.Attribute(nms_thresh, float)

    @torch.jit.script_method
    def forward(self, x):
        result = x.unsqueeze(0)

        box_coder = BoxCoder(2048, 1024)
        loc_preds, conf_preds, semantics, offsets_pred = self.net(result)

        boxes, labels, scores = box_coder.decode(loc_preds.squeeze(), F.softmax(conf_preds.squeeze(), dim=1),
                                                 self.score_thresh, self.nms_thresh)

        semantics_pred = semantics.argmax(dim=1)
        instance = helper.assign_pix2box(semantics_pred, offsets_pred, boxes, labels)

        return (boxes, labels, scores), (semantics_pred, offsets_pred), instance


if __name__ == '__main__':
    pred = Detector()

    # print(pred.graph_for(torch.randn(3, 224, 224)))
    with open('box2pix.txt', 'w+') as f:
        f.write(str(pred.graph))
    #    print(pred.graph)

    pred.save('predictor.pt')
