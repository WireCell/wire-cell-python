#!/usr/bin/env python3
'''
Things related to torch script
'''

import torch
import torch.fft


class DFT(torch.nn.Module):
    'Main interface to torch FFTs'
    def __init__(self):
        super(DFT, self).__init__()
        return

    # Match 1d, 1b and 2d IDFT API.
    def forward(self, arr: torch.Tensor, which:str, axis:int = -1) -> torch.Tensor:
        if which == "fwd1d":
            return torch.fft.fft(arr)
        if which == "inv1d":
            return torch.fft.ifft(arr)
        if which == "fwd2d":
            return torch.fft.fft2(arr)
        if which == "inv2d":
            return torch.fft.ifft2(arr)
        if which == "fwd1b":
            return torch.fft.fft(arr, dim=axis)
        if which == "inv1b":
            return torch.fft.ifft(arr, dim=axis)
        return arr

