import copy
import os
import torch

from mdir.learning.epoch_iteration.supervised_epoch import SupervisedEpoch
from mdir.components.data.dataset import initialize_dataset_loader
from mdir.tools import stats, tensors
from mdir.learning.network import SingleNetwork, MultiNetwork
from mdir.external.daan.core.path_resolver import resolve_path


class VisualDataset:

    decisive_criterion = None

    def __init__(self, params):
        """
        Outputs all images inside directory specified by image_dir into logger.
        Note this will not output any number, this only logs images for visual validation.

        params:
            -- data (dict): dataset/transforms/loader/mean_std to use for loading images to infer
            -- net_name (str, optional): specifies the name of a single network to search for deeply in multi network

        Example scenario description formats for rgb and grayscale images could be:

        criterion:
          type: visual
          data:
            dataset:
              name: InferImageList
              image_dir: data/val/day_night
            transforms: pil2np | totensor | normalize
            mean_std: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

        criterion:
          type: visual
          net_name: generator_X
          data:
            dataset:
              name: InferImageList
              image_dir: data/val/sketches
              mode: L
            transforms: anypil2np | totensor | normalize:False
            mean_std: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

        """
        self.net_name = params["net_name"] if "net_name" in params else None

        # Data init
        self.mean_std = copy.deepcopy(params["data"]["mean_std"])
        data = (os.listdir(resolve_path(params["data"]["dataset"]["image_dir"])),)
        self.loader = initialize_dataset_loader(data, "test", params.pop("data"), {"batch_size": 1})

        # Stats
        self.meter = stats.AverageMeter("Visual", len(self.loader), debug=True)
        self.resources = stats.ResourceUsage()

    def __call__(self, network, device, logger):
        network = self.get_infer_network(network)

        stopwatch = stats.StopWatch()

        print(">> Outputting images for visual validation...")
        dbg_data = {}
        network.eval()
        with torch.no_grad():
            for (i, (name, x)) in enumerate(self.loader):
                name = ".".join(name[0][-1].split(".")[:-1])
                dbg_data[name] = network(x.to(device))[-1]
                self.meter.update(i, None)

        dbg_data = tensors.to_device(tensors.detach(dbg_data), "cpu")
        loggeri = lambda *x: logger(-2, 1, *x)  # (iter -1) will be shown.
        for label, image in dbg_data.items():
            SupervisedEpoch._log_traindata_sample(image, loggeri, label, self.mean_std)

        stopwatch.lap("visual")
        logger(None, 1, "output", stopwatch.reset(), "scalar/time")
        self.resources.take_current_stats()

    def get_infer_network(self, network, defaults=("generator_X", "generator")):
        if isinstance(network, SingleNetwork):
            return network
        if self.net_name:
            if self.net_name in network:
                return network[self.net_name]
            for net in network.networks:
                if isinstance(net, MultiNetwork):
                    return self.get_infer_network(net)
        return network[defaults[0]] if defaults[0] in network else network[defaults[1]]
