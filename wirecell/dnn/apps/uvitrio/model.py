import torch
import torch.nn as nn
from wirecell.dnn.models import ViTUNetCrossView

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        input_shape=(1, 2560, 1500)
        self.model = ViTUNetCrossView(
            features=64,
            n_heads=4,
            n_blocks=2,
            ffn_features=128,
            embed_features=64,
            activ='gelu',
            norm='layer',
            input_shape=input_shape,
            output_shape=input_shape,
            unet_features_list=[32, 64, 128, 256, 512],  # 5 levels per paper
            unet_activ='relu',
            unet_norm='batch',  # Batch normalization per paper
            unet_downsample='maxpool',  # 2x2 max pooling per paper
            unet_upsample={'name': 'upsample', 'mode': 'bilinear'},  # Bilinear upsample per paper
            split_sizes=[800, 800, 960],
            split_dim=2,  # H dimension
            activ_output='sigmoid'
        )


        # self.unet = UNet(n_channels=3, n_classes=1,
        #                  batch_norm=True, bilinear=True, padding=True)

    def forward(self, x):
        # x = self.unet(x)
        return self.model(x)

