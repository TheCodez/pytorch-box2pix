import cv2

import torch
import torchvision.transforms as transforms

from models.box2pix import Box2Pix


class Predictor(object):

    def __init__(self, show_segmentation=True, show_labels=False, show_boxes=False):
        self.show_segmentation = show_segmentation
        self.show_labels = show_labels
        self.show_boxes = show_boxes

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = Box2Pix().to(self.device)
        self.net.eval()

        self.transform = self.get_transform()

    def get_transform(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform

    def run(self, image):
        result = image.copy()

        result = self.transform(result)
        result = result.unsqueeze(0)
        result = result.to(self.device)

        with torch.no_grad():
            loc_preds, conf_preds, semantics_pred, offsets_pred = self.net(result)

        if self.show_segmentation:
            result = self.add_segmentation_overlay(result, None)

        if self.show_boxes:
            result = self.add_boxes_overlay(result, None)

        if self.show_labels:
            result = self.add_overlay_classes(result, None)

        return result

    def add_boxes_overlay(self, image, predictions):

        for box in predictions:
            top_left, bottom_right = tuple(box[:2].tolist()), tuple(box[2:].tolist())
            image = cv2.rectangle(image, top_left, bottom_right, 0, 1)

        return image

    def add_segmentation_overlay(self, image, predictions):
        return image

    def add_overlay_classes(self, image, predictions):
        scores = [0]
        labels = [0]
        boxes = predictions

        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            cv2.putText(image, 'car', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

