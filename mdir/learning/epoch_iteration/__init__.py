from . import supervised_epoch, gan_epochs, cut_epochs, edges_epochs


EPOCH_ITERATIONS = {
    "SupervisedEpoch": supervised_epoch.SupervisedEpoch,
    "SupervisedCycleGanEpoch": gan_epochs.SupervisedCycleGanEpoch,
    "SupervisedCUTEpoch": cut_epochs.SupervisedCutEpoch,
    "SupervisedHEDGANEpoch": edges_epochs.SupervisedHedGanEpoch,
    "SupervisedHEDNGANEpoch": edges_epochs.SupervisedHedNGanEpoch,
}

def initialize_epoch_iteration(params, **kwargs):
    return EPOCH_ITERATIONS[params.pop("type")].initialize(params, **kwargs)
