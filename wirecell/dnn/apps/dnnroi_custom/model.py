import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet

class Network(nn.Module):

    def __init__(self, model_config):
        super().__init__()

        n_channels = model_config.get('n_channels', 3)
        print('Got n_channels', n_channels)

        self.unet = UNet(n_channels=int(n_channels), n_classes=1,
                         batch_norm=True, bilinear=True, padding=True)

    def forward(self, x):
        x = self.unet(x)
        return torch.sigmoid(x)

