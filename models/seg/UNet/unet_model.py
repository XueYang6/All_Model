""" Full assembly of the parts to form the complete network """
from .unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class R2UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, t=2, bilinear=False):
        super(R2UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (RRCNN_block(n_channels, 64, t=t))
        self.R2down1 = (RRCNN_Down(64, 128, t=t))
        self.R2down2 = (RRCNN_Down(128, 256, t=t))
        self.R2down3 = (RRCNN_Down(256, 512, t=t))
        factor = 2 if bilinear else 1
        self.R2down4 = (RRCNN_Down(512, 1024 // factor, t=t))
        self.up1 = (RRCNN_Up(1024, 512 // factor, t=t, bilinear=bilinear))
        self.up2 = (RRCNN_Up(512, 256 // factor, t=t, bilinear=bilinear))
        self.up3 = (RRCNN_Up(256, 128 // factor, t=t, bilinear=bilinear))
        self.up4 = (RRCNN_Up(128, 64, t=t, bilinear=bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.R2down1(x1)
        x3 = self.R2down2(x2)
        x4 = self.R2down3(x3)
        x5 = self.R2down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.R2down1 = torch.utils.checkpoint(self.R2down1)
        self.R2down2 = torch.utils.checkpoint(self.R2down2)
        self.R2down3 = torch.utils.checkpoint(self.R2down3)
        self.R2down4 = torch.utils.checkpoint(self.R2down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



