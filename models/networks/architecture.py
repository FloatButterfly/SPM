"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision

from models.networks.normalization import STYLE


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class STYLEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.noise_0 = ApplyNoise(fmiddle)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.noise_1 = ApplyNoise(fout)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=3, padding=1, bias=False)
            self.perconv_s = nn.Conv2d(opt.style_nc, opt.style_nc, kernel_size=1)
            self.noise_s = ApplyNoise(fout)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = STYLE(spade_config_str, fin, opt.semantic_nc, opt.style_nc, opt)
        self.norm_1 = STYLE(spade_config_str, fmiddle, opt.semantic_nc, opt.style_nc, opt)
        if self.learned_shortcut:
            self.norm_s = STYLE(spade_config_str, fin, opt.semantic_nc, opt.style_nc, opt)

        # define the 1*1 conv for style matrix for two style norm block
        self.perconv_0 = nn.Conv2d(opt.style_nc, opt.style_nc, kernel_size=1)
        self.perconv_1 = nn.Conv2d(opt.style_nc, opt.style_nc, kernel_size=1)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, st):
        if torch.isinf(st).any():
            raise AssertionError("inf in st")
        if torch.isnan(st).any():
            raise AssertionError("nan in st")

        x_s = self.shortcut(x, seg, st)
        st_0 = self.perconv_0(st)

        if torch.isinf(st_0).any():
            raise AssertionError("inf in st_0")
        if torch.isnan(st_0).any():
            raise AssertionError("nan in st_0")

        st_1 = self.perconv_1(st)

        dx = self.norm_0(x, seg, st_0)
        dx = self.conv_0(self.actvn(dx))

        # if self.training:
        dx = self.noise_0(dx)
        dx = self.norm_1(dx, seg, st_1)
        dx = self.conv_1(self.actvn(dx))

        # if self.training:
        dx = self.noise_1(dx)
        out = x_s + dx

        return out

    def shortcut(self, x, seg, st):
        if self.learned_shortcut:
            st_s = self.perconv_s(st)
            st_s = self.norm_s(x, seg, st_s)
            x_s = self.conv_s(st_s)
            x_s = self.noise_s(x_s)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
