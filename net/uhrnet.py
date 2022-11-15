import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBnReLU, UHRNet_W48

BN_MOMENTUM = 0.1

class UHRnet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'UHRNet_W18_Small'):
        super(UHRnet, self).__init__()
        if backbone == 'UHRNet_W18_Small':
            from .backbone import UHRNet_W18_Small
            self.backbone       = UHRNet_W18_Small()
            last_inp_channels   = int(279)

        if backbone == 'UHRNet_W48':
            from .backbone import UHRNet_W48
            self.backbone       = UHRNet_W48()
            last_inp_channels   = int(744)

        self.head = nn.Sequential()
        self.head.add_module("conv_1",
            ConvBnReLU(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.head.add_module("cls",
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
