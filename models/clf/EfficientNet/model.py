from torch import nn
from efficientnet_pytorch import EfficientNet


class EfficientNetClassification(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(EfficientNetClassification, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0',
                                                          in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        return self.efficient_net(x)
