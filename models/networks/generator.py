"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from models.networks.architecture import STYLEResnetBlock as STYLEResnetBlock
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, STYLE
from util import util


# spectralstylesyncbatch3x3

class STYLEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralstyleinstance3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('less', 'normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf  # 64

        self.sw, self.sh, self.num_upsampling_layers = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = STYLEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = STYLEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = STYLEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = STYLEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = STYLEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = STYLEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = STYLEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = STYLEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='bicubic')

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        elif opt.num_upsampling_layers == 'less':
            num_up_layers = 3
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh, num_up_layers

    def forward(self, input, st, img_dims=None):
        seg = input

        if img_dims is None:
            sh, sw = self.sh, self.sw
        else:
            factor = 2 ** self.num_upsampling_layers
            seg = util.pad_factor(seg, seg.size()[2:], factor)
            sh, sw = seg.size()[2] // factor, seg.size()[3] // factor

        if self.opt.label_type != 'edge':
            x = F.interpolate(seg, size=(sh, sw), mode='nearest')
        else:
            x = F.interpolate(seg, size=(sh, sw), mode='bilinear')

        x = self.fc(x)

        x = self.head_0(x, seg, st)
        x = self.up(x)

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_0(x, seg, st)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        if self.opt.num_upsampling_layers != 'less':
            x = self.G_middle_1(x, seg, st)
            x = self.up(x)

        x = self.up_0(x, seg, st)
        x = self.up(x)

        x = self.up_1(x, seg, st)
        x = self.up(x)

        x = self.up_2(x, seg, st)

        if self.opt.num_upsampling_layers != 'less':
            x = self.up(x)

        x = self.up_3(x, seg, st)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, st)

        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))

        x = F.tanh(x)

        if img_dims is not None:
            x = x[:, :, :img_dims[1], :img_dims[2]]

        return x
