"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


def style_extract(feature_map, mask, label_nc):
    x = feature_map
    mask = F.interpolate(mask, size=(x.size()[2], x.size()[3]), mode='nearest')
    mask = mask.round()
    style_matrix = torch.zeros((x.size()[0], x.size()[1], label_nc + 1, 1))
    region_list = np.unique(mask.cpu().numpy().astype(int))

    for i in region_list:
        indices = (mask == i)
        sum_indices = torch.sum(indices, dim=(2, 3), keepdim=True)
        ones = torch.ones_like(sum_indices)
        sum_indices = torch.where(sum_indices > 0, sum_indices, ones)
        sum_features = torch.sum(indices * x, dim=(2, 3), keepdim=True)
        style = sum_features / sum_indices
        style = style.view(style.size(0), style.size(1), 1, 1)

        style_matrix[:, :, i:i + 1, :] = style
    return style_matrix


class StyleEncoder(BaseNetwork):
    """
    encoder structure
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.label_nc = opt.label_nc
        final_nc = opt.style_nc
        ndf = 32
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.n_downsampling_layers = 2
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.Tconvlayer = norm_layer(nn.ConvTranspose2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, output_padding=1))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 8, final_nc, kw, stride=1, padding=pw))
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, mask):
        x = self.layer1(x)
        # img_dim = tuple(x.size()[1:])
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.Tconvlayer(self.actvn(x))
        x = self.layer4(self.actvn(x))
        style_matrix = style_extract(x, mask, self.label_nc)
        return style_matrix


class BasicBlockDown(nn.Module):
    def __init__(self, inplanes, outplanes, opt, nl_layer=None):
        super().__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.acvtv = nn.LeakyReLU(0.2, inplace=True)
        self.layer1 = norm_layer(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.layer2 = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shortcut_conv = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_0 = self.layer1(self.acvtv(x))
        x_0 = self.avg(self.layer2(self.acvtv(x_0)))
        x_s = self.shortcut_conv(self.avg(x))
        out = x_0 + x_s
        return out


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, opt, nl_layer=None):
        super().__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.acvtv = nn.LeakyReLU(0.2, inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer1 = norm_layer(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.layer2 = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True))
        self.shortcut_conv = norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.up(x)
        x_1 = self.layer1(self.acvtv(x))
        x_1 = self.layer2(self.acvtv(x_1))
        x_s = self.shortcut_conv(x)
        out = x_1 + x_s
        return out

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, True)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
