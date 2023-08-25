#
# Inspired from CUT loss computation:
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/cut_model.py
#


import torch

from . import gan_epochs
from mdir.tools import loss_value


class SupervisedCutEpoch(gan_epochs.SupervisedGanEpoch):

    def __init__(self, data_loader, criterion, mean_std):
        super().__init__(data_loader, criterion, mean_std)

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        netG = network.networks["generator_X"]
        netD = network.networks["discriminator_Y"]
        netF = network.networks["featdown"]

        # Forward
        real_X = batch_images.to(device)
        real_Y = batch_targets.to(device)
        real = torch.cat((real_X, real_Y), dim=0)
        fake = netG(real)
        fake_Y = fake[:real_X.size(0)]
        idt_Y = fake[real_X.size(0):]

        # (1) Discriminator step
        netD_loss, netD_loss_real, netD_loss_fake = self._unconditional_discriminator_step(
            optimizer["discriminator_Y"], netD, fake_Y, real_Y, device)

        # (2) Generator step
        netG_loss, netG_loss_GAN, netG_loss_NCE, netG_loss_IDT = self._generator_step_cut(
            optimizer["generator_X"], optimizer["featdown"], netG, netF, netD, real_X, fake_Y, real_Y, idt_Y, device)

        # (3) Log
        losses = [("total", (netG_loss + netD_loss).item()),
                  ("D_real", netD_loss_real.total.item()), ("D_fake", netD_loss_fake.total.item()),
                  ("G_gan", netG_loss_GAN.total.item()),
                  ("G_nce", netG_loss_NCE.total.item()),
                  ("G_idt", netG_loss_IDT.total.item())]
        losses += [("D_real_%s" % x, y.item()) for x, y in netD_loss_real.partial.items()]
        losses += [("D_fake_%s" % x, y.item()) for x, y in netD_loss_fake.partial.items()]
        losses += [("G_gan_%s" % x, y.item()) for x, y in netG_loss_GAN.partial.items()]
        losses += [("G_idt_%s" % x, y.item()) for x, y in netG_loss_IDT.partial.items()]
        losses += [("G_nce_%s" % x, y.item()) for x, y in netG_loss_NCE.partial.items()]
        return dict(losses), {"real_X": real_X[-1], "real_Y": real_Y[-1], "fake_Y": fake_Y[-1], "idt_Y": idt_Y[-1]}

    def _generator_step_cut(self, netG_optimizer, netF_optimizer, netG, netF, netD, real_X, fake_Y, real_Y, idt_Y, device):
        netG_optimizer.zero_grad()
        if netF_optimizer:
            netF_optimizer.zero_grad()
        pred_fake = netD.forward(fake_Y)

        criterion_adv, w_adv = self.criterion.losses["adversarial"], self.criterion.weights["adversarial"]
        criterion_idt, w_idt = self.criterion.losses["identity"], self.criterion.weights["identity"]
        criterion_nce, w_nce = self.criterion.losses["nce"], self.criterion.losses["nce"].weight

        netG_loss_GAN = w_adv * criterion_adv(pred_fake, is_target_real=True, device=device)

        netG_loss_NCE = w_nce * self.calculate_nce_loss(real_X, fake_Y, netG, netF) if w_nce > 0.0 else loss_value.ZERO

        if w_idt > 0.0 and w_nce > 0.0:
            netG_loss_IDT = w_idt * self.calculate_nce_loss(real_Y, idt_Y, netG, netF)
            netG_loss_NCE = (netG_loss_NCE + netG_loss_IDT) * 0.5
        else:
            netG_loss_IDT = loss_value.ZERO

        netG_loss = netG_loss_GAN.total + netG_loss_NCE.total
        netG_loss.backward()
        netG_optimizer.step()
        if netF_optimizer:
            netF_optimizer.step()
        return netG_loss, netG_loss_GAN, netG_loss_NCE, netG_loss_IDT

    def calculate_nce_loss(self, output, target, netG, netF):
        criterion = self.criterion.losses["nce"]
        nce_layers = criterion.nce_layers
        num_patches = criterion.num_patches

        feat_q = netG.forward(target, layers=nce_layers, encode_only=True)
        feat_k = netG.forward(output, layers=nce_layers, encode_only=True)
        feat_k_pool, sample_ids = netF.model(feat_k, num_patches=num_patches, patch_ids=None)
        feat_q_pool, _ = netF.model(feat_q, num_patches=num_patches, patch_ids=sample_ids)

        return criterion(feat_q_pool, feat_k_pool)
