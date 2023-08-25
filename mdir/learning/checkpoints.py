import os
import re
from pathlib import Path
import shutil
import torch

from daan.core.path_resolver import resolve_path
from daan.data.fs_driver import fs_driver

SUFFIX_NOTRAIN = "_notrain.pth"
SUFFIX_FROZEN = "_frozen.pth"
SUFFIX_EPOCH = "_epoch_%02d.pth"
SUFFIX_BEST_SO_FAR = "_bestsofar.pth"
SUFFIX_BEST = "_best.pth"
SUFFIX_LAST = "_last.pth"

FNAME_TRAINING = "learning_epoch_%02d.pth"


class Checkpoints:

    def __init__(self, directory, store_every, checkpoint_every, directory_epoch_regex):
        """
        Regularly store network and learning data.

        :param str directory: Base experiment directory. Checkpoints will be stored in epochs subfolder.
        :param int store_every: At the end of training, data of this epoch frequency and of the last
                epoch will be kept.
        :param int checkpoint_every: During training, checkpoint data to continue from with this
                epoch frequency (only the last checkpoint is kept during training)
        :param str directory_epoch_regex: Regex enabling to parse the epoch number from the directory
                basename, so that it is possible to continue from a finished experiment with less epochs.
                Regex must contain 3 groups, one for prefix, one for epoch number and one for postfix.
                Set to None to disable functionality.
        """
        self.directory = Path(resolve_path(directory)) / "epochs"
        self.store_every = store_every
        self.checkpoint_every = checkpoint_every
        self.directory_epoch_regex = directory_epoch_regex
        self.epoch_externally_loaded = -1

    def save_notrain(self, networks_state):
        # Handle helper variables
        if len(networks_state) > 1:
            networks_state["net"]["_network_names"] = [x for x in networks_state if x != "net"]

        if not self.directory.exists():
            os.makedirs(self.directory)

        # Save networks
        for key, state in networks_state.items():
            assert "/" not in key
            notrain_path = self.directory / (key + SUFFIX_NOTRAIN)
            torch.save(state, notrain_path)
            (self.directory / (key + SUFFIX_BEST)).symlink_to(key + SUFFIX_NOTRAIN)
            (self.directory / (key + SUFFIX_LAST)).symlink_to(key + SUFFIX_NOTRAIN)

    def save_epoch(self, networks_state, training_state, epoch, is_best, is_last):
        assert epoch >= 0
        epoch1 = epoch + 1
        is_checkpointed = (self.checkpoint_every > 0 and epoch1 % self.checkpoint_every == 0) or is_last
        is_stored = self.store_every > 0 and epoch1 % self.store_every == 0
        if is_checkpointed:
            last_checkpoint = epoch - (epoch1 % self.checkpoint_every or self.checkpoint_every)
            last_is_stored = self.store_every > 0 and (last_checkpoint + 1) % self.store_every == 0
            if last_checkpoint <= self.epoch_externally_loaded:
                last_checkpoint = -1

        if not self.directory.exists():
            os.makedirs(self.directory)

        # Handle helper variables
        if len(networks_state) > 1:
            networks_state["net"]["_network_names"] = [x for x in networks_state if x != "net"]

        # Save networks
        for key, state in networks_state.items():
            assert "/" not in key
            if state["frozen"]:
                # Does not change anymore, symlink only
                frozen_path = self.directory / (key + SUFFIX_FROZEN)
                if not frozen_path.exists():
                    torch.save(state, frozen_path)

            # Save epoch
            epoch_path = self.directory / (key + SUFFIX_EPOCH % epoch1)
            if is_checkpointed or is_stored:
                if state["frozen"]:
                    epoch_path.symlink_to(key + SUFFIX_FROZEN)
                else:
                    torch.save(state, epoch_path)

            # Symlink / save best & last
            shortcut_paths = []
            if is_best:
                shortcut_paths.append(self.directory / (key + SUFFIX_BEST_SO_FAR))
            if is_last:
                shortcut_paths.append(self.directory / (key + SUFFIX_LAST))
            for spath in shortcut_paths:
                if spath.exists():
                    spath.unlink()
                if state["frozen"]:
                    spath.symlink_to(key + SUFFIX_FROZEN)
                elif is_checkpointed or is_stored:
                    spath.symlink_to(key + SUFFIX_EPOCH % epoch1)
                else:
                    torch.save(state, spath)

        # Save training
        if is_checkpointed or is_stored:
            training_path = self.directory / (FNAME_TRAINING % epoch1)
            training_path_tmp = self.directory / ((FNAME_TRAINING % epoch1) + ".tmp")
            torch.save(training_state, training_path_tmp)
            training_path_tmp.rename(training_path)
            if is_checkpointed and last_checkpoint >= 0:
                (self.directory / (FNAME_TRAINING % (last_checkpoint+1))).unlink()

        # Remove unneeded networks
        for key, state in networks_state.items():
            best_path = self.directory / (key + SUFFIX_BEST_SO_FAR)
            if not best_path.exists():
                final_best = self.directory / (key + SUFFIX_BEST)
                if final_best.exists():
                    final_best.rename(best_path)
            # Remove previous if necessary
            if is_checkpointed and last_checkpoint >= 0 and not last_is_stored:
                previous_path = self.directory / (key + SUFFIX_EPOCH % (last_checkpoint+1))
                if previous_path == best_path.resolve():
                    best_path.unlink()
                    previous_path.rename(best_path)
                else:
                    previous_path.unlink()

            # Handle last
            if is_last and best_path.exists():
                best_path.rename(self.directory / (key + SUFFIX_BEST))

    @staticmethod
    def _load_epoch_network(directory, suffix):
        network_state = {
            "net": torch.load(directory / ("net" + suffix), map_location=lambda storage, location: storage)
        }

        # Embedded networks
        assert "net" not in network_state["net"].get("_networks_included", {})
        network_state.update(network_state["net"].pop("_networks_included", {}))

        # External networks
        for name in network_state["net"].pop("_network_names", []):
            assert name not in network_state
            epoch_path = directory / (name + suffix)
            network_state[name] = torch.load(epoch_path, map_location=lambda storage, location: storage)
        return network_state

    @staticmethod
    def _load_epoch_training(directory, suffix):
        return torch.load(directory / suffix)

    def load_latest_epoch(self, nepochs):
        # Search for previously stored epochs
        if self.directory.exists():
            for epoch in reversed(range(nepochs)):
                epoch1 = epoch + 1
                training_path = self.directory / (FNAME_TRAINING % epoch1)
                if training_path.exists():
                    network = self._load_epoch_network(self.directory, SUFFIX_EPOCH % epoch1)
                    training = self._load_epoch_training(self.directory, FNAME_TRAINING % epoch1)
                    print(">>", "Loading stored epoch %02d" % epoch1)
                    return network, training

        # Search for previous experiments
        if self.directory_epoch_regex:
            match = re.search(self.directory_epoch_regex, str(self.directory))
            assert match, "directory_epoch_regex does not match with learning.checkpoints.directory"
            for epoch1 in reversed(range(1, int(match.group(2)))):
                directory = Path("%s%s%s" % (match.group(1), epoch1, match.group(3)))
                if not directory.exists():
                    continue
                training_path = directory / (FNAME_TRAINING % epoch1)
                if not training_path.exists():
                    continue

                # Verify that epochs were parsed correctly
                assert (directory / ("net" + SUFFIX_LAST)).resolve() == (directory / ("net" + SUFFIX_EPOCH % epoch1)).resolve(), \
                    "{} != {}".format((directory / ("net" + SUFFIX_LAST)), (directory / ("net" + SUFFIX_EPOCH % epoch1)))

                # Retain the best network
                if not self.directory.exists():
                    os.makedirs(self.directory)
                state = torch.load(directory / ("net" + SUFFIX_EPOCH % epoch1), map_location=lambda storage, location: storage)
                for name in ["net"] + state.get("_network_names", []):
                    shutil.copy(str((directory / (name + SUFFIX_BEST)).resolve()),
                                str(self.directory / (name + SUFFIX_BEST_SO_FAR)))
                # Retain blobs
                if (self.directory / "blobs").exists():
                    shutil.rmtree(str(self.directory / "blobs"))
                shutil.copytree(str(directory / "blobs"), str(self.directory / "blobs"))

                # Load from that folder
                network = self._load_epoch_network(directory, SUFFIX_EPOCH % epoch1)
                training = self._load_epoch_training(directory, FNAME_TRAINING % epoch1)
                self.epoch_externally_loaded = epoch1-1
                print(">>", "Loading epoch %02d from experiment %s" % (epoch1, directory))
                return network, training

        return None

    @classmethod
    def load_network(cls, directory):
        if directory is None:
            return None
        directory = resolve_path(directory)
        if Path(directory).is_dir():
            return cls._load_epoch_network(Path(directory), SUFFIX_BEST)

        with fs_driver(directory).open() as handle:
            checkpoint = torch.load(handle, map_location=lambda storage, location: storage)

        assert "net" not in checkpoint.get("_networks_included", {})
        return {"net": checkpoint, **checkpoint.pop("_networks_included", {})}
