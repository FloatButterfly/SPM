"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from torch import nn
from models.spm_model import SPMModel


class End2EndTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.model = SPMModel(opt).cuda(opt.local_rank)
        print(self.model)
        self.loss_item = None

        if len(opt.gpu_ids) > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
            )

            self.model_on_one_gpu = self.model.module
            self.model_on_one_gpu = self.model_on_one_gpu.cuda()
        else:
            self.model_on_one_gpu = self.model
            torch.cuda.set_device(0)

        self.decode_image = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr


    def run_one_step(self, data, lmbda=-1):

        self.optimizer_G.zero_grad()
        self.loss, self.bpp, self.decode_image, self.loss_item = self.model_on_one_gpu(data, mode="generator",
                                                                                               lmbda=lmbda)
        self.loss.backward()
        self.optimizer_G.step()


    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated


    def run_discriminator_one_step(self, data, regularize=False):
        self.optimizer_D.zero_grad()
        self.d_losses, self.d_loss_item = self.model_on_one_gpu(data, mode='discriminator')
        self.d_losses.backward()
        self.optimizer_D.step()

    def get_latest_losses(self):
        return {**self.loss, **self.loss_item}

    def get_latest_generated(self):
        return {**self.decode_image, **self.decode_semantic}

    # def update_learning_rate(self, epoch):
    #     self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr


    def initiate_learning_rate(self, opt):
        if self.opt.no_TTUR:
            new_lr_G = opt.lr
            new_lr_D = opt.lr
        else:
            new_lr_G = opt.lr / 2
            new_lr_D = opt.lr * 2

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('initiate learning rate for new rate: %f -> %f' % (new_lr_G, new_lr_D))
        self.old_lr=opt.lr

