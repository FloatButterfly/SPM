"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from models.networks.sync_batchnorm import SynchronizedBatchNorm2d


def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class STYLE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, style_nc, opt):
        super().__init__()
        self.label_nc = label_nc
        self.label_type = opt.label_type
        assert config_text.startswith('style')
        parsed = re.search('style(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )

        self.mlp_gamma_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_gamma_s = nn.Conv2d(style_nc, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta_s = nn.Conv2d(style_nc, norm_nc, kernel_size=ks, padding=pw)
        self.w_gamma = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.w_beta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x, seg, st):


        normalized = self.param_free_norm(x)
        B, H, W = x.size(0), x.size(2), x.size(3)
        C, T = st.size(1), st.size(2)

        if self.label_type != 'edge':
            segmap = F.interpolate(seg, size=x.size()[2:],
                                   mode='nearest')

            st_ = st.repeat(1, 1, 1, H * W).reshape(B, C, T, H, W)
            seg_ = segmap.unsqueeze(1).repeat(1, C, 1, 1, 1).reshape(B, C, T, H, W)
            style_map = torch.sum(st_ * seg_, dim=2)
        else:
            segmap = F.interpolate(seg, size=x.size()[2:], mode='bilinear')
            style_map = st.expand(st.size(0), st.size(1), x.size(2), x.size(3))


        gamma_s = self.mlp_gamma_s(style_map)
        beta_s = self.mlp_beta_s(style_map)
        actv = self.mlp_shared(segmap)
        gamma_o = self.mlp_gamma_o(actv)
        beta_o = self.mlp_beta_o(actv)
        gamma = self.w_gamma * gamma_o + (1 - self.w_gamma) * gamma_s
        beta = self.w_beta * beta_o + (1 - self.w_beta) * beta_s
        out = normalized * (1 + gamma) + beta

        return out
