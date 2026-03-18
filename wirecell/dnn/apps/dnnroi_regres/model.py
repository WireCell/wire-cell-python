import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.unet = UNet(n_channels=1, n_classes=2,
                         batch_norm=True, bilinear=True, padding=True)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        x = self.unet(x)
        # x[:, 0] = torch.sigmoid(x[:, 0])
        # x[:, 1] = self.leaky_relu(x[:, 1])
        return torch.stack(
            [
                torch.sigmoid(x[:, 0]),
                # self.leaky_relu(x[:, 1])
                x[:,1]
            ], dim=1
        )

