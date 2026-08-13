import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet

import logging
log = logging.getLogger("wirecell.dnn")

class Network(nn.Module):

    def __init__(self, **cfg):
        super().__init__()

        n_channels = cfg.get('n_channels', 3)
        log.info(f'dnnroi_custom network: n_channels={n_channels}')

        self.unet = UNet(n_channels=int(n_channels), n_classes=1,
                         batch_norm=True, bilinear=True, padding=True)

    def forward(self, x):
        x = self.unet(x)
        return x
