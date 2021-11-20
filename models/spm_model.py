"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import models.networks as networks
from lpips_pytorch import lpips
from util import util
from .EntropyBottleneck import TextureEntropyBottleneck
from .networks import *

tensor_kwargs = {"dtype": torch.float32, "device": torch.device("cuda:0")}


class SPMModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.flag = True if self.opt.semantic_nc % 2 else False
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.texture_entropy = TextureEntropyBottleneck(self.opt.style_nc, opt).to(**tensor_kwargs)
        self.texture_entropy.init_weights(opt.init_type, opt.init_variance)

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionCD = networks.GANLoss('ls', tensor=self.FloatTensor, opt=self.opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            if self.opt.k_vgg:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            init_loss = self.FloatTensor(1).fill_(0).cuda(self.opt.local_rank)
            self.G_losses = {"mse": init_loss, "L1": init_loss, "GAN_Feat": init_loss,
                             "GAN": init_loss,  "latent": init_loss,
                             "VGG": init_loss}
            self.D_losses = {"D_fake": init_loss, "D_real": init_loss}


    def val_forward(self, data):
        input_semantic = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantic)
        style_matrix = self.netE(real_image, input_semantic).cuda(self.opt.local_rank)

        if self.opt.binary_quant:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix1 = q_style_matrix1 * (2 ** self.opt.qp_step)
        else:
            q_style_matrix1, style_z_hat1, texture_length1, latents_decor1 = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix1 = q_style_matrix1 * self.opt.qp_step

        tex_bpp = texture_length1 / (real_image.numel() / real_image.size(1))
        decode_image = self.netG(input_semantics_onehot, q_style_matrix1)

        return decode_image, tex_bpp, q_style_matrix1

    def transfer_forward(self, data, style_code):
        input_semantic = data['label'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantic)
        if self.opt.binary_quant:
            q_style_code, style_z_hat, texture_length, _ = self.texture_entropy(
                style_code / (2 ** self.opt.qp_step))
            q_style_code = q_style_code * (2 ** self.opt.qp_step)
        else:
            q_style_code, style_z_hat, texture_length, _ = self.texture_entropy(style_code / self.opt.qp_step)
            q_style_code = q_style_code * self.opt.qp_step
        transfer_img = self.netG(input_semantics_onehot, q_style_code)
        return transfer_img

    def test_forward(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        img_dims = tuple(real_image.size()[1:])

        n_encoder_downsamples = self.netE.n_downsampling_layers
        factor = 2 ** n_encoder_downsamples
        real_image = util.pad_factor(real_image, real_image.size()[2:], factor)
        input_semantics = util.pad_factor(input_semantics, input_semantics.size()[2:], factor)
        input_semantics_onehot_pad = self.to_onehot(input_semantics)  # after padding

        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)  # (bs,512,19,1)

        if self.opt.binary_quant:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2 ** self.opt.qp_step)
        else:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix = q_style_matrix * self.opt.qp_step

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
        decode_image = self.netG(input_semantics_onehot, q_style_matrix, img_dims)

        return real_image, decode_image, tex_bpp, q_style_matrix, latents_decor

    def encode_forward(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        style_matrix = self.netE(real_image, input_semantics)  # (bs,512,19,1)
        return style_matrix


    def recons_prepare(self, data):
        input_semantics = data['label'].cuda(self.opt.local_rank)
        real_image = data['image'].cuda(self.opt.local_rank)
        input_semantics_onehot = self.to_onehot(input_semantics)  # unpadding
        style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)
        if torch.isnan(style_matrix).any():
            raise AssertionError("nan in style matrix")
        elif self.opt.binary_quant:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / (2 ** self.opt.qp_step))
            q_style_matrix = q_style_matrix * (2 ** self.opt.qp_step)
        else:
            q_style_matrix, style_z_hat, texture_length, latents_decor = self.texture_entropy(
                style_matrix / self.opt.qp_step)
            q_style_matrix = q_style_matrix * self.opt.qp_step

        tex_bpp = texture_length / (real_image.numel() / real_image.size(1))  # nbits / B*H*W
        decode_image = self.netG(input_semantics_onehot, q_style_matrix)

        restyle_matrix = self.netE(decode_image, input_semantics).cuda(self.opt.local_rank)
        if self.opt.binary_quant:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / (2 ** self.opt.qp_step))
            q_restyle_matrix = q_restyle_matrix * (2 ** self.opt.qp_step)
        else:
            q_restyle_matrix, _, _, _ = self.texture_entropy(restyle_matrix / self.opt.qp_step)
            q_restyle_matrix = q_restyle_matrix * self.opt.qp_step

        return tex_bpp, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix

    def forward(self, data, mode='generator', regularize=False, style_code=None, lmbda=-1):
        if mode == "val":
            decode_image, tex_bpp, q_style_matrix = self.val_forward(data)
            return decode_image, tex_bpp, q_style_matrix
        elif mode == "test":
            real_image, decode_image, tex_bpp, q_style_matrix, noise_scale = self.test_forward(data)
            return real_image, decode_image, tex_bpp, q_style_matrix, noise_scale
        elif mode == "transfer":
            transfer_img = self.transfer_forward(data, style_code)
            return transfer_img
        elif mode == "encode":
            style_matrix = self.encode_forward(data)
            return style_matrix
        else:
            tex_bpp, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix = self.recons_prepare(
                data)
            if mode == "generator":
                loss, G_losses = self.compute_generator_loss(input_semantics_onehot, real_image, decode_image,
                                                             q_style_matrix, q_restyle_matrix)

                loss = loss + self.opt.lmbda * tex_bpp
                return loss, tex_bpp, decode_image, G_losses
            elif mode == 'discriminator':
                loss, D_losses = self.compute_discriminator_loss(input_semantics_onehot, real_image, decode_image)
                return loss, D_losses

    def latent_regularizer(self, image1, image2, latent_code1, latent_code2):
        pass

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        G_params += list(self.netE.parameters())
        G_params += list(self.texture_entropy.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2  # 默认使用TTUR，G 0.0001，D 0.0004

        optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(beta1, beta2))

        if opt.isTrain:
            D_params = list(self.netD.parameters())
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def load_network(net, label, epoch, opt):
        save_filename = '%s_net_%s.pth' % (epoch, label)
        print(save_filename)
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        save_path = os.path.join(save_dir, save_filename)
        weights = torch.load(save_path)
        net.load_state_dict(weights)
        return net

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netE, 'E', epoch, self.opt)
        util.save_network(self.texture_entropy, 'texture_entropy', epoch, self.opt)
        optimizer = {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }
        torch.save(optimizer, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pkl'))

    ############################################################################
    # Private helper methods
    ############################################################################
    def _param_num(self):
        semantic_codec_sum = sum(p.numel() for p in self.semantic_codec.parameters() if p.requires_grad)
        entroy_sum = sum(p.numel() for p in self.texture_entropy.parameters() if p.requires_grad)
        print("semantic codec trainable parameters: %d\n" % semantic_codec_sum)
        print("texture entropy model trainable parameters: %d\n" % entroy_sum)

    def initialize_networks(self, opt):
        print("********", opt.semantic_nc)
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt)

        if not opt.isTrain or opt.continue_train:

            self.texture_entropy = util.load_network(self.texture_entropy, 'texture_entropy', opt.which_epoch, opt)
            print("load texture entropy success")
            self.netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            print("load netG success")
            self.netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            print("load netE success")
            if opt.GANmode:
                self.netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                print("load netD success")
            if self.opt.isTrain:
                self.optimizer_G, self.optimizer_D = self.create_optimizers(opt)
            if opt.resume:
                save_dir = os.path.join(opt.checkpoints_dir, opt.name)
                optim_ckpt = torch.load(os.path.join(save_dir, 'optimizer.pkl'))
                self.optimizer_G.load_state_dict(optim_ckpt['optimizer_G'])
                self.optimizer_D.load_state_dict(optim_ckpt['optimizer_D'])
        return netG, netD, netE


    def to_onehot(self, label):
        label_map = label.long().cuda(self.opt.local_rank)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_().cuda(self.opt.local_rank)
        input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        return input_semantics_onehot

    def preprocess_input(self, data):
        # move to GPU and change data types
        # data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda(self.opt.local_rank)
            data['image'] = data['image'].cuda(self.opt.local_rank)

        # create one-hot label map
        if self.opt.label_type == 'edge':
            input_semantics_onehot = data['label']
        else:
            label_map = data['label'].long().cuda(self.opt.local_rank)
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_().cuda(self.opt.local_rank)
            input_semantics_onehot = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics_onehot = torch.cat((input_semantics_onehot, instance_edge_map), dim=1)

        return data['label'], input_semantics_onehot, data['image']

    def compute_generator_loss(self, input_semantics_onehot, real_image, decode_image, q_style_matrix, q_restyle_matrix):
        # G_losses = {}
        if self.opt.k_gan:
            pred_fake, pred_real = self.discriminate(
                input_semantics_onehot, decode_image, real_image)

            self.G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                     for_discriminator=False)
        if self.opt.k_latent:
            self.G_losses['latent'] = self.criterionL1(q_style_matrix, q_restyle_matrix)
        if self.opt.k_L1:
            self.G_losses['L1'] = self.criterionL1(real_image, decode_image)

        if self.opt.k_feat:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0).cuda(self.opt.local_rank)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionL1(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.k_feat / num_D
            self.G_losses['GAN_Feat'] = GAN_Feat_loss
        if self.opt.k_mse:
            self.G_losses['mse'] = torch.mean((real_image - decode_image) ** 2) * 255 ** 2
        if self.opt.k_vgg:
            self.G_losses['VGG'] = self.criterionVGG(decode_image, real_image)

        loss = self.opt.k_latent * self.G_losses['latent'] + \
               self.opt.k_feat * self.G_losses['GAN_Feat'] + \
               self.opt.k_vgg * self.G_losses['VGG'] + \
               self.opt.k_gan * self.G_losses['GAN'] + \
               self.opt.k_L1 * self.G_losses['L1']

        return loss, self.G_losses

    def compute_discriminator_loss(self, input_semantics_onehot, real_image, decode_image, real_image2=None,
                                   decode_image2=None, regularize=False):
        decode_image = decode_image.detach()
        decode_image.requires_grad_()
        if self.opt.k_gan:
            pred_fake, pred_real = self.discriminate(
                input_semantics_onehot, decode_image, real_image)

            self.D_losses['D_fake'] = self.criterionGAN(pred_fake, False,
                                                        for_discriminator=True)
            self.D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                        for_discriminator=True)
        loss = sum(self.D_losses.values()).mean()
        return loss, self.D_losses

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_fake(self, input_semantics, input_semantics_onehot, real_image):
        if self.opt.label_type == 'edge':
            style_matrix = self.netE(real_image).cuda(self.opt.local_rank)
        else:
            style_matrix = self.netE(real_image, input_semantics).cuda(self.opt.local_rank)
        fake_image = self.netG(input_semantics_onehot, style_matrix)

        return fake_image


    def discriminate(self, input_semantics_onehot, fake_image, real_image):
        fake_concat = torch.cat([input_semantics_onehot, fake_image], dim=1)
        real_concat = torch.cat([input_semantics_onehot, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
