import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet

class Network(nn.Module):

    def __init__(self, model_config):
        super().__init__()

        self.regres_only = int(model_config.get('regres_only', 0))
        print('Regres only?', self.regres_only)

        self.unet = UNet(n_channels=1, n_classes=(1 if self.regres_only else 2),
                         batch_norm=True, bilinear=True, padding=True)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.unet(x)

        if self.regres_only:
            # return torch.sigmoid(x)
            return x
        else:
            return torch.stack(
                [
                    torch.sigmoid(x[:, 0]),
                    x[:,1]
                ], dim=1
            )

