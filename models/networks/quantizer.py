import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def forward(self, x, is_training, offset=0, noise_scale=None):
        if is_training:
            y = x + torch.empty_like(x).uniform_(-0.5, 0.5)
        else:
            y = torch.round(x - offset) + offset
        return y
