import torch

from mdir.components.optim.criterion.compound_losses import DiscriminatorLoss
from . import supervised_epoch
from mdir.tools import gan_image_pool


class SupervisedGanEpoch(supervised_epoch.SupervisedEpoch):
    """Base epoch iteration for all kinds of gan training."""

    def __init__(self, data_loader, criterion, mean_std):
        super().__init__(data_loader, criterion, mean_std, batch_average=None, fakebatch=False)

    def _unconditional_discriminator_step(self, netD_optimizer, netD, fake_Y, real_Y, device):
        return self._discriminator_step(netD_optimizer, netD, None, fake_Y, real_Y, device)

    def _conditional_discriminator_step(self, netD_optimizer, netD, real_X, fake_Y, real_Y, device):
        return self._discriminator_step(netD_optimizer, netD, real_X, fake_Y, real_Y, device)

    def _discriminator_step(self, netD_optimizer, netD, real_X, fake_Y, real_Y, device):
        netD_optimizer.zero_grad()
        pred_real, pred_fake = self.__netD_forward(netD, real_Y, fake_Y, real_X)
        criterion, w = self.criterion.losses["adversarial"], self.criterion.weights["adversarial"]
        netD_loss_real = w * criterion(pred_real, is_target_real=True, device=device)
        netD_loss_fake = w * criterion(pred_fake, is_target_real=False, device=device)
        netD_loss = (netD_loss_real + netD_loss_fake) * 0.5
        netD_loss.backward()
        netD_optimizer.step()
        return netD_loss, netD_loss_real, netD_loss_fake

    def __netD_forward(self, netD, real_Y, fake_Y, real_X=None):
        if real_X:
            pred_real = netD.forward(torch.cat([real_X, real_Y], dim=1))
            pred_fake = netD.forward(torch.cat([real_X, fake_Y], dim=1).detach())
        else:
            pred_real = netD.forward(real_Y)
            pred_fake = netD.forward(fake_Y.detach())  # Detached to not update generator
        return pred_real, pred_fake

    def _generator_step(self, netG_optimizer, netD, real_X, fake_Y, real_Y, device):
        netG_optimizer.zero_grad()
        pred_fake = netD.forward(torch.cat([real_X, fake_Y], dim=1))
        criterion_adv, w_adv = self.criterion.losses["adversarial"], self.criterion.weights["adversarial"]
        criterion_reg, w_reg = self.criterion.losses["regularization"], self.criterion.weights["regularization"]
        netG_loss_GAN = w_adv * criterion_adv(pred_fake, is_target_real=True, device=device)
        netG_loss_reg = w_reg * criterion_reg(fake_Y, real_Y)
        netG_loss = netG_loss_GAN + netG_loss_reg
        netG_loss.backward()
        netG_optimizer.step()
        return netG_loss, netG_loss_GAN, netG_loss_reg

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        raise NotImplementedError("Attempted to optimize abstract GAN. Choose different GAN epoch iteration.")


#
# Inspired from CycleGAN loss computation:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
#

class SupervisedCycleGanEpoch(SupervisedGanEpoch):

    def __init__(self, data_loader, criterion, mean_std, pool_size):
        super().__init__(data_loader, criterion, mean_std)
        self.fake_X_pool = gan_image_pool.GanImagePool(pool_size)
        self.fake_Y_pool = gan_image_pool.GanImagePool(pool_size)

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        netG_X = network.networks["generator_X"]
        netG_Y = network.networks["generator_Y"]
        netD_X = network.networks["discriminator_X"]
        netD_Y = network.networks["discriminator_Y"]

        # Forward pass:
        # We are optimizing the mapping X <-> Y, where
        #   X is input domain
        #   Y is output domain

        real_X = batch_images.to(device)
        real_Y = batch_targets.to(device)
        fake_Y = netG_X.forward(real_X)
        rec_X = netG_Y.forward(fake_Y)
        fake_X = netG_Y.forward(real_Y)
        rec_Y = netG_X.forward(fake_X)

        # Backward pass:

        # (1) Generators step
        netD_X.requires_grad = False
        netD_Y.requires_grad = False
        netG_X_loss = self._generator_step_cyclegan(optimizer["generator_X"], self.criterion.loss_G_X,
                                                    netD_X, real_X, fake_Y, rec_X, device)
        netG_Y_loss = self._generator_step_cyclegan(optimizer["generator_Y"], self.criterion.loss_G_Y,
                                                    netD_Y, real_Y, fake_X, rec_Y, device)
        # Calculate gradients and update there, because G_X and G_Y are shared together for both rec_X and rec_Y
        netG_X_loss.backward()
        netG_Y_loss.backward()
        optimizer["generator_X"].step()
        optimizer["generator_Y"].step()

        # (2) Discriminators step
        netD_X.requires_grad = True
        netD_Y.requires_grad = True
        netD_X_loss = self._discriminator_step_cyclegan(optimizer["discriminator_X"], self.criterion.loss_D_X,
                                                        netD_X, real_Y, fake_Y, device, self.fake_X_pool)
        netD_Y_loss = self._discriminator_step_cyclegan(optimizer["discriminator_Y"], self.criterion.loss_D_Y,
                                                        netD_Y, real_X, fake_X, device, self.fake_Y_pool)

        loss = netG_X_loss.total + netG_Y_loss.total + netD_X_loss.total + netD_Y_loss.total

        # (3) Log
        losses = [("netG_X_%s" % x, y) for x, y in netG_X_loss.item()]
        losses += [("netG_Y_%s" % x, y) for x, y in netG_Y_loss.item()]
        losses += [("netD_X_%s" % x, y) for x, y in netD_X_loss.item()]
        losses += [("netD_Y_%s" % x, y) for x, y in netD_Y_loss.item()]
        return {"total": loss.item(), **dict(losses)}, \
               {"real_X": real_X[-1], "fake_Y": fake_Y[-1], "rec_X": rec_X[-1],
                "real_Y": real_Y[-1], "fake_X": fake_X[-1], "rec_Y": rec_Y[-1]}

    def _generator_step_cyclegan(self, netG_optimizer, loss_G, netD, real, fake, rec, device):
        netG_optimizer.zero_grad()
        pred_fake = netD.forward(fake)
        pred_target = DiscriminatorLoss.get_target_tensor(pred_fake, is_target_real=True, device=device)
        loss = loss_G.forward({"adversarial": pred_fake, "cycle": rec},
                              {"adversarial": pred_target, "cycle": real})
        # loss.backward() and optimizer.step() are performed later
        return loss

    def _discriminator_step_cyclegan(self, netD_optimizer, loss_D, netD, real, fake, device, fake_pool):
        netD_optimizer.zero_grad()
        fake = fake_pool.query(fake)  # To reduce model oscillation, take 1/2 previous fake images pool
        pred_real = netD.model.forward_multi(real)
        pred_fake = netD.model.forward_multi(fake.detach())  # Detach because of not backprop to generator
        loss_real = loss_D(pred_real, is_target_real=True, device=device)
        loss_fake = loss_D(pred_fake, is_target_real=False, device=device)
        loss = (loss_real + loss_fake) * 0.5

        loss.backward()
        netD_optimizer.step()
        return loss
