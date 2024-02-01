import torch.nn as nn
from torchvision.models.detection import (maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights,
                                          MaskRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskRCNNResNet50(nn.Module):
    def __init__(self, n_classes):
        super(MaskRCNNResNet50, self).__init__()
        self.maskrcnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.n_classes = n_classes
        self.in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
        self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, n_classes)

    def forward(self, images, targets=None):
        if self.training:
            return self.maskrcnn(images, targets)  # In training mode, pass in both images and targets
        else:
            return self.maskrcnn(images)  # In inference mode, only the image is passed in
