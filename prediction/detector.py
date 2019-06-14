import torch
import torch.nn as nn
import torch.nn.functional as F

from model.box2pix import box2pix
from utils import box_utils
from utils.box_coder import BoxCoder


class Detector(nn.Module):

    def __init__(self, width=2048, height=1024, score_thresh=0.6, nms_thresh=0.5):
        super(Detector, self).__init__()
        self.box_coder = BoxCoder(width, height)

        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]

        self.net = box2pix(pretrained=False)
        self.net.eval()

        self.img_size = (width, height)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def normalize(self, image):
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def forward(self, x):
        image = self.normalize(x)

        loc_preds, conf_preds, semantics_pred, offsets_pred = self.net(image)

        boxes, labels, scores = self.box_coder.decode(loc_preds.squeeze(), F.softmax(conf_preds.squeeze(), dim=1),
                                                      self.score_thresh, self.nms_thresh)

        boxes = box_utils.resize_boxes(boxes, image)
        instance = self.box_coder.assign_box2pix(semantics_pred, offsets_pred, boxes, labels)

        return (boxes, labels, scores), semantics_pred, offsets_pred, instance


if __name__ == '__main__':
    pred = Detector()

    # print(pred.graph_for(torch.randn(3, 224, 224)))
    with open('box2pix.txt', 'w+') as f:
        f.write(str(pred.graph))
    #    print(pred.graph)

    pred.save('predictor.pt')
