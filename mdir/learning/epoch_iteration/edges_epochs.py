from . import gan_epochs


class SupervisedHedGanEpoch(gan_epochs.SupervisedGanEpoch):
    def __init__(self, data_loader, criterion, mean_std):
        super().__init__(data_loader, criterion, mean_std)

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        netG = network.networks["generator_X"]
        netD = network.networks["discriminator_Y"]
        netH = network.networks["detector"]

        # Forward
        real_X = batch_images.to(device)
        real_Y = batch_targets.to(device)
        fake_Y = netG(real_X)
        fake_E = netH(fake_Y)
        real_E = netH(real_X)

        # (1) Discriminator step
        netD_loss, netD_loss_real, netD_loss_fake = self._unconditional_discriminator_step(
            optimizer["discriminator_Y"], netD, fake_Y, real_Y, device)

        # (2) Generator step
        netG_loss, netG_loss_GAN, netG_loss_HED = self._generator_hed_step(
            optimizer["generator_X"], netD, fake_Y, fake_E, real_E, device)

        # (3) Log
        losses = [("total", (netG_loss + netD_loss).item()),
                  ("D_real", netD_loss_real.total.item()),
                  ("D_fake", netD_loss_fake.total.item()),
                  ("G_gan", netG_loss_GAN.total.item()),
                  ("G_hed", netG_loss_HED.item())]
        losses += [("D_real_%s" % x, y.item()) for x, y in netD_loss_real.partial.items()]
        losses += [("D_fake_%s" % x, y.item()) for x, y in netD_loss_fake.partial.items()]
        losses += [("G_gan_%s" % x, y.item()) for x, y in netG_loss_GAN.partial.items()]
        dbg_data = {"real_X": real_X[-1], "real_Y": real_Y[-1], "fake_Y": fake_Y[-1], "real_E": real_E[-1],
                    "fake_E": fake_E[-1]}
        return dict(losses), dbg_data

    def _generator_hed_step(self, netG_optimizer, netD, fake_Y, fake_E, real_E, device):
        netG_optimizer.zero_grad()
        pred_fake = netD.forward(fake_Y)

        criterion_adv, w_adv = self.criterion.losses["adversarial"], self.criterion.weights["adversarial"]
        criterion_hed, w_hed = self.criterion.losses["edge"], self.criterion.weights["edge"]

        netG_loss_GAN = w_adv * criterion_adv(pred_fake, is_target_real=True, device=device)
        netG_loss_HED = w_hed * criterion_hed(fake_E, real_E)  # Edge consistency loss
        netG_loss = netG_loss_GAN.total + netG_loss_HED

        netG_loss.backward()
        netG_optimizer.step()
        return netG_loss, netG_loss_GAN, netG_loss_HED


class SupervisedHedNGanEpoch(SupervisedHedGanEpoch):
    def __init__(self, data_loader, criterion, mean_std):
        super().__init__(data_loader, criterion, mean_std)

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        netG = network.networks["generator_X"]
        netD = network.networks["discriminator_Y"]
        netH_student = network.networks["detector"]
        netH_teacher = network.networks["detector_frozen"]

        # Forward
        real_X = batch_images.to(device)
        real_Y = batch_targets.to(device)
        fake_Y = netG(real_X)

        # (1) Discriminator step (same as in HEDGAN)
        netD_loss, netD_loss_real, netD_loss_fake = self._unconditional_discriminator_step(
            optimizer["discriminator_Y"], netD, fake_Y, real_Y, device)

        # (2-3) Generator step (new in HED-N-GAN)
        # Optimization needs to be done separate, if the generator needs sigmoid edges,
        # but the detector distillation is performed before sigmoid.

        # (2) Detector step
        # Perform detector first, otherwise it backwards through the graph a second time,
        # but the saved intermediate results have already been freed.
        netE_loss, netE_loss_real, netE_loss_fake = self._detector_step(
            optimizer["detector"], netH_student, netH_teacher, real_X, fake_Y)

        # (3) Generator step
        fake_E = netH_student(fake_Y)
        real_E, real_E_check = netH_teacher(real_X), netH_student(real_X)
        netG_loss, netG_loss_GAN, netG_loss_HED = self._generator_hed_step(
            optimizer["generator_X"], netD, fake_Y, fake_E, real_E, device)

        # (4) Log
        losses = [("total", (netG_loss + netD_loss).item()),
                  ("D_real", netD_loss_real.total.item()),
                  ("D_fake", netD_loss_fake.total.item()),
                  ("G_gan", netG_loss_GAN.total.item()),
                  ("G_hed", netG_loss_HED.item()),
                  ("E_real", netE_loss_real.item()),
                  ("E_fake", netE_loss_fake.item())]
        losses += [("D_real_%s" % x, y.item()) for x, y in netD_loss_real.partial.items()]
        losses += [("D_fake_%s" % x, y.item()) for x, y in netD_loss_fake.partial.items()]
        losses += [("G_gan_%s" % x, y.item()) for x, y in netG_loss_GAN.partial.items()]
        dbg_data = {"real_X": real_X[-1], "real_Y": real_Y[-1], "fake_Y": fake_Y[-1],
                    "real_E": real_E[-1], "fake_E": fake_E[-1], "real_E_check": real_E_check[-1]}
        return dict(losses), dbg_data

    def _detector_step(self, optimizer, netH_student, netH_teacher, real_X, fake_Y):
        optimizer.zero_grad()
        criterion_hed, w_hed = self.criterion.losses["hed"], self.criterion.weights["hed"]

        target_M = netH_teacher.forward(real_X, no_sigmoid=True)
        real_M = netH_student.forward(real_X, no_sigmoid=True)
        netE_loss_real = w_hed * criterion_hed(real_M, target_M)

        fake_M = netH_student.forward(fake_Y.detach(), no_sigmoid=True)
        netE_loss_fake = w_hed * criterion_hed(fake_M, target_M)

        netE_loss = netE_loss_real + netE_loss_fake
        netE_loss.backward()
        optimizer.step()
        return netE_loss, netE_loss_real, netE_loss_fake
