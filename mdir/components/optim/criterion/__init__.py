from . import base_losses, compound_losses, cirlosses


CRITERIA = {
    "l1": base_losses.L1Loss,
    "mse": base_losses.MSELoss,
    "bce": base_losses.BCELoss,
    "bce_with_logits": base_losses.BCEWithLogitsLoss,
    "contrastive": cirlosses.ContrastiveLoss,
    "contrastive_multidesc": cirlosses.ContrastiveLossMultipleDescriptors,
    "triplet": cirlosses.TripletLoss,
    "cycle_loss": compound_losses.CycleLoss,
    "discriminator_loss": compound_losses.DiscriminatorLoss,
    "loss_set": compound_losses.LossSet,
    "multihead_loss": compound_losses.MultiheadLoss,
    "combination_loss": compound_losses.CombinationLoss,
    "multilayer_patchnce_loss": compound_losses.MultilayerPatchNCELoss,
}

def initialize_criterion(params):
    if not params:
        return None

    return CRITERIA[params.pop("loss")](**params)
