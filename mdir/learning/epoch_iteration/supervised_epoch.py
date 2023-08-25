import copy
import torch

from mdir.tools import loss_value, tensors
from mdir.tools.stats import StopWatch
from mdir.tools.utils import indent
from mdir.components.optim.criterion import initialize_criterion
from mdir.components.data.dataset import initialize_dataset_loader


class SupervisedEpoch:

    LOG_TRAINDATA_SAMPLE_EVERY = 5

    def __init__(self, data_loader, criterion, mean_std, *, batch_average, fakebatch):
        self.data_loader = data_loader
        self.criterion = criterion
        self.mean_std = mean_std
        self.epoch = None
        self.batch_average = batch_average
        self.fakebatch = fakebatch

    @classmethod
    def initialize(cls, params_epoch, data, params_data, default_criterion, network):
        data_key = params_epoch.pop("data")
        net_defaults = network.network_params.runtime.get("data", {})
        data_params = {**net_defaults, **params_data[data_key]}
        data_loader = initialize_dataset_loader(data, "train", copy.deepcopy(data_params), {"shuffle": True})

        criterion_section = params_epoch.pop("criterion")
        if criterion_section == "default":
            if default_criterion is None:
                raise ValueError("Criterion cannot be 'default' when default criterion is not specified")
            criterion = default_criterion
        else:
            criterion = initialize_criterion(criterion_section)

        return cls(data_loader=data_loader, criterion=criterion, mean_std=data_params["mean_std"], **params_epoch)

    def steps(self, epoch):
        self.epoch = epoch
        return self

    def _loss_computation(self, network, images, target, device):
        """Return tuple (loss, output, target)"""
        output = network.forward(images)
        target = tensors.to_device(target, device)
        loss = self.criterion(output, target)
        return loss, output, target

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        """Perform a single optimization step. Meant to be overriden by children classes."""
        assert self.criterion.reduction in {"mean", "sum"}, self.criterion.reduction
        criterion_mean_reduction = self.criterion.reduction == "mean"

        optimizer.zero_grad()

        if self.fakebatch:
            # Save gpu memory by backprop after each image
            cumloss = loss_value.ZERO
            batch_size = len(batch_images)
            image, target = None, None
            for image, target in zip(batch_images, batch_targets):
                loss, output, target = self._loss_computation(network, image, target, device)

                # Handle batch average
                if self.batch_average > criterion_mean_reduction: # already_mean=False, batch_average=True
                    loss /= batch_size
                elif self.batch_average < criterion_mean_reduction: # already_mean=True, batch_average=False
                    loss *= batch_size

                loss.backward()
                cumloss = cumloss + loss.item()

            # Single step for a batch
            optimizer.step()

            # Report averaged
            if not self.batch_average:
                cumloss /= batch_size

            dbg_data = {"input": image, "output": output, "target": target}
            if isinstance(cumloss, loss_value.MultiValue):
                return dict(cumloss), dbg_data
            return {"total": cumloss}, dbg_data

        # Regular step
        loss, batch_output, batch_targets = self._loss_computation(network, batch_images, batch_targets, device)

        # Handle batch average
        if self.batch_average > criterion_mean_reduction: # already_mean=False, batch_average=True
            loss /= len(batch_images)
        elif self.batch_average < criterion_mean_reduction: # already_mean=True, batch_average=False
            loss *= len(batch_images)

        loss.backward()
        optimizer.step()

        # Report averaged
        cumloss = loss.item()
        if self.batch_average is not None and not self.batch_average:
            cumloss /= len(batch_images)

        dbg_data = {"input": batch_images[-1], "output": batch_output[-1], "target": batch_targets[-1]}
        if isinstance(cumloss, loss_value.MultiValue):
            return dict(cumloss), dbg_data
        return {"total": cumloss}, dbg_data

    @staticmethod
    def _log_parameter_weights(network, logger):
        with torch.no_grad():
            for train_data in network.train_data():
                logger(train_data["key"], train_data["data"], train_data["dtype"])

    @staticmethod
    def _log_traindata_sample(image, logger, label, mean_std):
        dbg_data = {}
        with torch.no_grad():
            if isinstance(image, dict):
                for key, value in image.items():
                    dbg_data[key] = {"dtype": "text", "data": value.tolist()}
            else:
                if not isinstance(image, list):
                    image = [image]
                if not isinstance(image[0], torch.Tensor):
                    # Cannot process at the moment
                    return
                shape = image[0].shape
                # Log only 1-channel or 3+channel images
                if len(shape) < 1 or len(shape) == 2 or min(shape[-2:]) < 20 \
                        or (3 <= len(shape) <= 4 and (shape[len(shape)-3] != len(mean_std[0])
                                                      and shape[len(shape)-3] != 1)):
                    # Cannot process at the moment
                    return

                mean, std = torch.Tensor(mean_std[0]), torch.Tensor(mean_std[1])
                # Greyscale (output) image together with RGB:
                if ((shape[0] == 1 and len(shape) == 3) or (shape[1] == 1 and len(shape) == 4)) and mean.shape[0] > 1:
                    mean, std = torch.Tensor([0.0]), torch.Tensor([1.0])

                for j, img in enumerate(image):
                    if len(img.size()) == 4:
                        img = img[0]
                    nchans = img.size(0)
                    if nchans >= 3:
                        dbg_data["image%s.rgb" % j] = {"dtype": "image:rgb", "data": img[:3] * std[:3,None,None] + mean[:3,None,None]}
                        # Skip other channels
                        if j >= 3:
                            continue
                    for k in range(3 if nchans >= 3 else 0, nchans):
                        dbg_data["image%s.chan%s" % (j, k+1)] = {"dtype": "image:gray", "data": img[k] * std[k,None,None] + mean[k,None,None]}
                        # Skip other channels
                        if j >= 3:
                            break

            logger("data/%s" % label, dbg_data, "blob")

    def prepare_offtheshelf(self, network, device):
        pass

    def prepare_epoch(self, network, device, logger, stopwatch):
        train_loader = self.data_loader

        # Prepare epoch in dataset
        if hasattr(train_loader.dataset, "prepare_epoch"):
            metadata = train_loader.dataset.prepare_epoch(network, device)
            stopwatch.lap("prepare_data")
            if metadata:
                logger(None, len(train_loader), "learning/data_mining", metadata, "scalar/loss")

    def iterate(self, network, optimizer, device, logger, *, include_debug_data=False):
        train_loader = self.data_loader
        stopwatch = StopWatch()

        network.eval()

        self.prepare_epoch(network, device, logger, stopwatch)
        logger(None, len(train_loader), "learning/prepare_epoch", stopwatch.reset(include_total=False), "scalar/time")

        if self.epoch == 0:
            self._log_parameter_weights(network, logger=lambda *x: logger(-1, len(train_loader), *x))

        network.train()

        for i, (batch_images, batch_targets) in enumerate(train_loader):
            stopwatch.lap("prepare_data")
            losses, dbg_data = self._optimization_step(network, optimizer, device, batch_images, batch_targets)
            stopwatch.lap("process_batch")
            logger(i, len(train_loader), "learning/loss", losses, "scalar/loss")

            # Take stats
            if i == len(train_loader)-1:
                self._log_parameter_weights(network, logger=lambda *x: logger(i, len(train_loader), *x))
            if (i == len(train_loader)-1 and (self.epoch+1) % self.LOG_TRAINDATA_SAMPLE_EVERY == 0) \
                    or (i == 0 and self.epoch == 0):
                loggeri = lambda *x: logger(i, len(train_loader), *x)
                dbg_data = tensors.to_device(tensors.detach(dbg_data), "cpu")
                for key, sample in dbg_data.items():
                    self._log_traindata_sample(sample, loggeri, key, self.mean_std)

            if include_debug_data:
                # Copying tensors to cpu every iteration has performance cost
                dbg_data = tensors.to_device(tensors.detach(dbg_data), "cpu")
            else:
                # Yielding gpu tensors with grad tree every iteration leads to gpu memory leak in some circumstances
                dbg_data = {}

            yield losses, dbg_data

            stopwatch.lap("take_statistics")
            logger(i, len(train_loader), "learning/iteration", stopwatch.reset(include_total=False), "scalar/time")

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    dataset: {indent(str(self.data_loader.dataset))}
    criterion: {indent(str(self.criterion))}
    fakebatch: {self.fakebatch}
    batch_average: {self.batch_average}
)"""
