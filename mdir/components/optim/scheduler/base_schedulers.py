import math
from torch.optim import lr_scheduler


class VoidScheduler:

    def step(self):
        pass


def init_void_scheduler(_optimizer, _last_epoch, _nepochs):
    return VoidScheduler()

def init_lambda_scheduler(optimizer, last_epoch, nepochs, fixed_ratio):
    """First, have fixed lr, then, decay it linearly to zero"""
    # Fixed ratio is e.g. 0.5
    def lambda_rule(epoch):
        return 1 - max(0, epoch + 1 - fixed_ratio*nepochs) / float((1-fixed_ratio)*nepochs + 1)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

def init_gamma_scheduler(optimizer, last_epoch, _nepochs, gamma):
    # Gamma is e.g. 0.99 ~ exp(-0.01)
    if isinstance(gamma, str) and gamma.startswith("exp(") and gamma[-1] == ")":
        gamma = math.exp(float(gamma[len("exp("):-1]))

    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def init_lambda_scheduler_p2p(optimizer, last_epoch, nepochs, n_epochs_decay):
    """ Linear lr decay from original cycleGAN implementation [1]:
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.

    Parameters:
        optimizer      -- the optimizer of the network
        last_epoch     -- current epoch count, -1 by default during init
        nepochs        -- n_epochs + n_epochs_decay in [1], n_epochs=100 by default
        n_epochs_decay -- original parameter from [1], n_epochs_decay=100 by default

    [1]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    n_epochs = nepochs - n_epochs_decay
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)


# Initialization

BASE_SCHEDULERS = {
    "const": init_void_scheduler,
    "lambda": init_lambda_scheduler,
    "lambda_p2p": init_lambda_scheduler_p2p,
    "gamma": init_gamma_scheduler,
}

def initialize_base_scheduler(optimizer, last_epoch, nepochs, params):
    return BASE_SCHEDULERS[params.pop('algorithm')](optimizer, last_epoch, nepochs, **params)
