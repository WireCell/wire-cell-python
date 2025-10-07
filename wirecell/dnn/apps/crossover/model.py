import torch
import torch.nn as nn
from wirecell.dnn.models.unet import UNet
from wirecell.raygrid import crossover
class Network(nn.Module):

    def __init__(self, nfeatures=None):
        super().__init__()
        self.unets = [
                UNet(n_channels=3, n_classes=1,
                     batch_norm=True, bilinear=True, padding=True)
                for i in range(3)
        ]
        self.crossover_term = crossover.CrossoverTerm()

    def forward(self, x):
        '''
        Input data is assumed to be of shape (nbatch, nfeatures, nticks, nchannels=3)
        '''
        print(x)
        x = self.unet(x)
        return torch.sigmoid(x)

