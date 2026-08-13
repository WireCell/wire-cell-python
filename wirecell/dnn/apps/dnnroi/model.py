import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet

class Network(nn.Module):

    def __init__(self, **cfg):
        # This app takes no model config; **cfg is accepted so obj_with_config
        # can call every app's Network the same way.
        super().__init__()
        self.unet = UNet(n_channels=3, n_classes=1,
                         batch_norm=True, bilinear=True, padding=True)

    def forward(self, x):
        x = self.unet(x)
        return torch.sigmoid(x)

