import os
import copy
import time
import warnings
from collections import namedtuple
import abc
from PIL import Image
import torch

from daan.data.fs_driver import fs_driver
from daan.core.path_resolver import resolve_path
from ..components.model import network, weight_initialization
from ..components.data.wrapper import initialize_wrappers
from ..tools.utils import indent
from ..tools import tensors

warnings.filterwarnings("always", "CirNetwork", DeprecationWarning)


class Network(abc.ABC):

    TRAIN = "train"
    EVAL = "eval"

    def __init__(self, frozen, model=None):
        self.stage = None
        self.frozen = frozen
        self.model = model
        if frozen:
            self.eval()

    def __call__(self, image):
        """Method used for inference (due to compatibility with cirtorch project)"""
        return self.forward(image)

    @abc.abstractmethod
    def forward(self, image):
        """Method to be used for training. Comapred to __call__ can return a dictionary with
            different values (e.g. in multihead setup)"""

    @staticmethod
    def initialize_wrappers(wrappers, device):
        if isinstance(wrappers, dict):
            assert wrappers.keys() == {"train", "eval"}, wrappers.keys()
            return {x: initialize_wrappers(wrappers[x], device) for x in wrappers}

        return {x: initialize_wrappers(wrappers, device) for x in ["train", "eval"]}

    def train(self):
        if not self.frozen:
            self.model.train()
            self.stage = Network.TRAIN
        return self

    def eval(self):
        self.model.eval()
        self.stage = Network.EVAL
        return self

    def freeze(self, net="net"):
        assert net == "net"
        self.frozen = True
        self.eval()
        return self

    def parameters(self, optimizer_opts, net="net"):
        assert net == "net"
        if self.frozen:
            return []
        if hasattr(self.model, "parameter_groups"):
            return self.model.parameter_groups(optimizer_opts)
        return self.model.parameters()

    def set_meta(self, meta):
        self.meta = meta
        if self.model:
            self.model.meta = meta

    # Debug data functions

    def train_data(self):
        return [{"key": "net/params", "dtype": "weight/param", "data": dict(self.model.named_parameters())}]

    def const_data(self):
        acc = []
        graph = self._generate_network_graph()
        if isinstance(graph, tuple):
            for i, graphi in enumerate(graph):
                acc.append({"key": "network_graph", "dtype": "blob", "data": {"output%s" % i: {"dtype": "image:rgb", "data": graphi}}})
        elif graph is not None:
            acc.append({"key": "network_graph", "dtype": "blob", "data": {"net": {"dtype": "image:rgb", "data": graph}}})
        return acc

    def _generate_network_graph(self):
        return None

#
# Standalone networks
#

class SingleNetwork(Network):

    """
    NetworkParams fields

    model: [model parameters dict]
    runtime:
      frozen: [whether the network is learned]
      wrappers: [stage-dependent or independent wrapper definition]
      model: [parameters that will be propagated to contained model]
      data:
        mean_std: [2d array with default mean and std]
        transforms: [default transforms]
    """
    NetworkParams = namedtuple("NetworkParams", ["model", "runtime"])

    def __init__(self, model, network_params, device, frozen):
        self.meta = model.meta if model.meta else {}
        if "model" in network_params.runtime:
            model.runtime = network_params.runtime["model"]

        self.network_params = network_params
        self.wrappers = self.initialize_wrappers(network_params.runtime.get("wrappers", ""), device)
        super().__init__(network_params.runtime.get("frozen", False) or frozen, model.to(device))
        self.device = device

        # Limit to supported keys
        assert not network_params.runtime.keys() - {"data", "wrappers", "frozen", "model"}, \
            network_params.runtime.keys() - {"data", "wrappers", "frozen", "model"}
        assert not network_params.runtime.get("data", {}).keys() - {"mean_std", "transforms"}, \
            network_params.runtime.get("data", {}).keys() - {"mean_std", "transforms"}

    def forward(self, image, **params):
        return self.wrappers[self.stage](image, self.forward_batch, outputmodel=self.model, tensor_params=params)

    def forward_batch(self, images, **params):
        if images is None:
            return None
        if isinstance(images, list):
            return [self.model(tensors.as_tensor(x), **params) if x is not None else None for x in images]
        return self.model(tensors.as_tensor(images), **params)

    @classmethod
    def initialize(cls, params, device):
        # Initialize model
        path = params.pop("path", None)
        if not path:
            network_params = cls.NetworkParams(params.pop("model"),
                                               params.pop("runtime"))
            model = network.initialize_model(copy.deepcopy(network_params.model))
            # Initialize weights
            init = params.pop("initialize")
            if init and isinstance(init, str):
                # Initialize weights from path
                with fs_driver(init).open() as handle:
                    model.load_state_dict(torch.load(handle))
            elif init and init["weights"] != "default":
                # Initialize weights randomly
                weights = init.pop("weights")
                seed = init.pop("seed")
                torch.manual_seed(seed if seed is not None else time.time())
                model.apply(weight_initialization.initialize_weights(weights, init))
        else:
            # Pretrained model
            print(">> Loaded net from %s" % path)
            path = resolve_path(path)
            with fs_driver(path).open() as handle:
                checkpoint = torch.load(handle, map_location=lambda storage, location: storage)
            # Handle runtime inheritance
            runtime = params.pop("runtime")
            if runtime == "load_from_checkpoint":
                runtime = checkpoint["network_params"]["runtime"]
            else:
                runtime = {x: y if y != "load_from_checkpoint" else checkpoint["network_params"]["runtime"][x] for x, y in runtime.items()}
            # Initialize
            network_params = cls.NetworkParams(checkpoint["network_params"]["model"], runtime)
            model = network.initialize_model(copy.deepcopy(network_params.model))
            model.load_state_dict(checkpoint['model_state'])
            # Can be optionally provided
            params.pop("initialize", None)
            if "model" in params:
                params_model = params.pop("model")
                assert params_model == checkpoint["network_params"]["model"],\
                    "{} != {}".format(params_model, checkpoint["network_params"]["model"])

        assert not params, params.keys()

        return cls(model, network_params, device=device, frozen=False)

    def overlay_params(self, new_params, device=None):
        if not new_params:
            return self

        new_params["runtime"]["frozen"] = True
        network_params = self.NetworkParams(self.network_params.model,
                                            new_params.pop("runtime"))
        assert not new_params

        model = self.model
        if "model" in network_params.runtime:
            model = model.copy_with_runtime(network_params.runtime['model'])

        return self.__class__(model, network_params, device or self.device, frozen=True)

    def overlay_model(self, new_model, device=None):
        return self.__class__(new_model, self.network_params, device or self.device, frozen=True)

    #
    # Load and save
    #

    def state_dict(self):
        return {
            "net": {
                "type": self.__class__.__name__,
                "frozen": self.frozen,
                "network_params": self.network_params._asdict(),
                "model_state": self.model.state_dict(),
            }
        }

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        assert state_dict.keys() == {"net"}, state_dict.keys()
        checkpoint = state_dict["net"]
        assert checkpoint.keys() == {"type", "frozen", "network_params", "model_state"}, checkpoint.keys()
        network_params = cls.NetworkParams(**checkpoint["network_params"])

        assert checkpoint["type"] == cls.__name__, checkpoint["type"]
        if params is not None and "path" not in params:
            del params["initialize"]
            assert network_params._asdict() == params, "%s != %s" % (str(network_params._asdict()), str(params))

        model = network.initialize_model(copy.deepcopy(network_params.model))
        model.load_state_dict(checkpoint['model_state'])

        if runtime:
            network_params.runtime.update(runtime)

        return cls(model, network_params, device=device, frozen=checkpoint["frozen"])

    #
    # Utils
    #

    def _generate_network_graph(self):
        if getattr(self.model, '_disable_graphviz', False):
            return None

        # A bit of a hack
        import random

        model = network.initialize_model(copy.deepcopy(self.network_params.model))
        if "model" in self.network_params.runtime:
            model.runtime = copy.deepcopy(self.network_params.runtime["model"])

        in_channels = self.meta.get("in_channels", 3)
        if in_channels < 20:
            x_in = torch.zeros(1, in_channels, 512, 512, requires_grad=True)
        else:
            x_in = torch.zeros(in_channels, 1, requires_grad=True)
        fname = '/tmp/jenicto2.network_graph.%s' % random.randint(0, 1000000)
        y_pred = model(x_in)
        if isinstance(y_pred, tuple):
            return tuple(self._generate_graph_for_input(x_in, x, model, fname) for x in y_pred)
        return self._generate_graph_for_input(x_in, y_pred, model, fname)

    @staticmethod
    def _generate_graph_for_input(x_in, y_pred, model, fname):
        # A bit of a hack
        import imageio
        from torchviz.dot import make_dot

        make_dot(y_pred, params=dict(list(model.named_parameters()) + [('x', x_in)])).render(fname, cleanup=True)
        try:
            img = imageio.imread(fname+".png")
        except Image.DecompressionBombError:
            img = None
        os.remove(fname+".png")
        return img

    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.network_params._asdict().items())
        nice_wrappers = "\n" + "".join("    %s: %s,\n" % (x, indent(str(y))) for x, y in self.wrappers.items())

        return \
f"""{self.__class__.__name__} (
    meta: {self.meta}
    model: {indent(str(self.model))}
    network_params: {{{indent(nice_params)}}}
    wrappers: {{{indent(nice_wrappers)}}}
)"""


class SingleNetworkLink(SingleNetwork):
    """ Network having model linked to reference network with LinkedNetworkSet.

    NetworkParams fields

    runtime:
      frozen: [whether the network is learned]
      wrappers: [stage-dependent or independent wrapper definition]
      data:
        mean_std: [2d array with default mean and std]
        transforms: [default transforms]
    """
    NetworkParams = namedtuple("NetworkParams", ["runtime"])

    def __init__(self, model, network_params, device, frozen):
        super().__init__(model, network_params, device, frozen)
        self.reference_network = None  # placeholder network

    @classmethod
    def initialize(cls, params, device):
        model = network.Identity()  # placeholder model
        network_params = cls.NetworkParams(params.pop("runtime"))
        return cls(model, network_params, device=device, frozen=False)

    def overlay_params(self, new_params, device=None):
        if not new_params:
            return self

        new_params["runtime"]["frozen"] = True
        network_params = self.NetworkParams(new_params.pop("runtime"))
        assert not new_params

        return self.__class__(self.model, network_params, device or self.device, frozen=True)

    #
    # Load and save
    #

    def state_dict(self):
        return {
            "net": {
                "type": self.__class__.__name__,
                "frozen": self.frozen,
                "network_params": self.network_params._asdict(),
            }
        }

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        assert state_dict.keys() == {"net"}, state_dict.keys()
        checkpoint = state_dict["net"]
        assert checkpoint.keys() == {"type", "frozen", "network_params"}, checkpoint.keys()
        network_params = cls.NetworkParams(**checkpoint["network_params"])

        assert checkpoint["type"] == cls.__name__, checkpoint["type"]
        if params is not None:
            assert network_params._asdict() == params, "%s != %s" % (str(network_params._asdict()), str(params))

        if runtime:
            network_params.runtime.update(runtime)

        model = network.Identity()  # placeholder model

        return cls(model, network_params, device=device, frozen=checkpoint["frozen"])

    def _generate_network_graph(self):
        return None


class CirNetwork(SingleNetwork):

    def __init__(self, *args, **kwargs):
        warnings.warn("CirNetwork is deprecated and will be removed, use SingleNetwork", DeprecationWarning)
        super().__init__(*args, **kwargs)


class GlobalLocalNetwork(SingleNetwork):
    """Network that can operate in both local-descriptor and global-descriptor setup"""

    SCALES = {
        "ss": [1],
        "msdelf": [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25],
    }

    def __call__(self, image):
        return super().forward(image)

    def forward(self, images, **params):
        forward_train = lambda x: self.model.forward_train(tensors.as_tensor(x))
        return self.wrappers[self.stage](images, forward_train, outputmodel=self.model)

    def forward_global(self, image, **kwargs):
        """Return global descriptor"""
        if 'scales' in kwargs and isinstance(kwargs['scales'], str):
            kwargs['scales'] = self.SCALES[kwargs['scales']]

        model = lambda x: self.model.forward_global(tensors.as_tensor(x), **kwargs)
        return self.wrappers[self.stage](image, model)

    def forward_local(self, image, **kwargs):
        """Return local descriptors"""
        if 'scales' in kwargs and isinstance(kwargs['scales'], str):
            kwargs['scales'] = self.SCALES[kwargs['scales']]

        return self.model.forward_local(tensors.as_tensor(image), **kwargs)

    def features(self, image):
        return self.model.features_attentions(image, scales={"ss": [1], "msdelf": [1.0]})[0][0]

    def copy_excluding_dim_reduction(self):
        return self.overlay_model(self.model.copy_excluding_dim_reduction())

    @property
    def dim_reduction(self):
        return self.model.dim_reduction

    @property
    def runtime(self):
        return self.model.runtime


#
# Multi-net networks
#

class MultiNetwork(Network):
    """
    Abstract class providing basic multi-network functionality.
    """

    def __init__(self, networks, network_order, frozen):
        assert len(networks) == len(network_order)
        self.networks = networks
        self.network_order = network_order
        super().__init__(frozen)

        assert set(network_order) == networks.keys()

    def __contains__(self, key):
        if key in self.networks:
            return True

        for net in self.network_order:
            if isinstance(self.networks[net], MultiNetwork) and key in self.networks[net]:
                return True

        return False

    def __getitem__(self, key):
        if key in self.networks:
            return self.networks[key]

        for net in self.network_order:
            if isinstance(self.networks[net], MultiNetwork) and key in self.networks[net]:
                return self.networks[net][key]

        raise KeyError("'%s'" % key)

    @staticmethod
    def _initialize_subnetworks(networks, device):
        return {x: NETWORKS[networks[x].pop("type")].initialize(networks[x], device) for x in networks}

    def train(self):
        for net in self.network_order:
            self.networks[net].train()
        self.stage = Network.TRAIN
        return self

    def eval(self):
        for net in self.network_order:
            self.networks[net].eval()
        self.stage = Network.EVAL
        return self

    def freeze(self, net=None):
        if net is not None:
            self[net].freeze()
            return self

        for net in self.network_order:
            self.networks[net].freeze()
        self.frozen = True
        return self

    def parameters(self, optimizer_opts, net=None):
        return self._parameters_with_groups(optimizer_opts, net=net)

    def _parameters_with_groups(self, optimizer_opts, net=None, parameter_groups=None):
        if net is not None:
            return self[net].parameters(optimizer_opts)

        acc = []
        for net in self.network_order:
            params = self.networks[net].parameters(optimizer_opts)
            if not isinstance(params, list):
                params = [{"params": params}]
            if parameter_groups:
                for key, val in parameter_groups.get(net, {}).items():
                    for parami in params:
                        parami[key] = parami.get(key, optimizer_opts[key]) * val
            acc += params
        return acc

    def named_parameters(self, optmizer_opts, net=None):
        if net is not None:
            return self[net].named_parameters(optmizer_opts)
        raise NotImplementedError

    def _overlay_subnetworks(self, new_params, device):
        diff = set(self.network_order) - set(new_params.keys())
        assert not diff, diff

        acc = {}
        for net in self.network_order:
            acc[net] = self.networks[net]
            if net in new_params:
                acc[net] = acc[net].overlay_params(new_params[net], device)

        return acc

    #
    # Load and save
    #

    def state_dict(self):
        network_hierarchy = {}
        state = {}
        for net in self.network_order:
            netstate = self.networks[net].state_dict()
            netstate[net] = netstate.pop("net")
            # Enforce zero overlap
            intersection = set(state.keys()).intersection(netstate.keys())
            assert not intersection, intersection
            # Nesting
            network_hierarchy[net] = [x for x in netstate if x != net]
            state.update(netstate)

        state["net"] = {
            "type": self.__class__.__name__,
            "frozen": self.frozen,
            "network_order": self.network_order,
            "network_hierarchy": network_hierarchy,
        }
        return state

    @staticmethod
    def _initialize_subnetworks_from_state(network_hierarchy, state_dict, device, params, net_runtimes):
        if params is not None:
            assert set(params.keys()) == network_hierarchy.keys(), (params.keys(), network_hierarchy.keys())

        acc = {}
        for net in network_hierarchy:
            netparams = params[net] if params is not None else None
            netstate = {x: state_dict[x] for x in network_hierarchy[net]}
            netstate["net"] = state_dict[net]
            ntype = netparams.pop("type") if netparams else state_dict[net]["type"]
            nruntime = net_runtimes[net] if net_runtimes else None
            acc[net] = NETWORKS[ntype].initialize_from_state(netstate, device, netparams, nruntime)
        return acc

    #
    # Utils
    #

    def train_data(self):
        acc = []
        for net in self.network_order:
            train_data = self.networks[net].train_data()
            acc += [{**x, "key": x["key"].replace("net/", net+"/")} for x in train_data]
        return acc

    def const_data(self):
        acc = []
        graphs = {}
        for net in self.network_order:
            for const_data in self.networks[net].const_data():
                if const_data["key"] == "network_graph":
                    if "net" in const_data["data"]:
                        graphs[net] = const_data["data"].pop("net")
                    graphs.update({"%s/%s" % (net, x): y for x, y in const_data["data"].items()})
                else:
                    acc.append({**const_data, "key": "%s/%s" % (net, const_data["key"])})
        if graphs:
            acc.append({"key": "network_graph", "dtype": "blob", "data": graphs})
        return acc


class NetworkSet(MultiNetwork):
    """
    Provides a minimal functionality on the top of MultiNetwork for a set of networks without inner
    structure. Suitable as a container to multiple networks, if they are used separately (does not
    provide forward()), e.g. by epoch_iteration.
    """

    NetworkParams = namedtuple("NetworkParams", ["runtime"])

    def __init__(self, networks, sequence, frozen):
        super().__init__(networks, sequence, frozen)
        self.network_params = self.NetworkParams({})

    def forward(self, image):
        raise NotImplementedError("The structure of networks is undefined")

    @classmethod
    def initialize(cls, params, device):
        networks = cls._initialize_subnetworks(params, device)
        return cls(networks, list(sorted(networks.keys())), frozen=False)

    def overlay_params(self, new_params, device):
        if not new_params:
            return self

        networks = self._overlay_subnetworks(new_params, device)
        return self.__class__(networks, self.network_order, frozen=True)

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        checkpoint = state_dict.pop("net")
        assert checkpoint["type"] == cls.__name__
        assert checkpoint.keys() == {"type", "frozen", "network_order", "network_hierarchy"}, checkpoint.keys()
        assert set(checkpoint["network_order"]) == checkpoint['network_hierarchy'].keys()

        assert not runtime
        networks = cls._initialize_subnetworks_from_state(checkpoint["network_hierarchy"], state_dict, device, params, None)
        return cls(networks, checkpoint["network_order"], frozen=checkpoint["frozen"])

    def __repr__(self):
        nice_networks = ""
        for net in self.network_order:
            nice_networks += f"        {net}: {indent(str(self.networks[net]), 2)}\n"

        return \
f"""{self.__class__.__name__} (
    network_order: {self.network_order}
    networks: {{
{nice_networks}
    }}
)"""


class SequentialNetwork(MultiNetwork):

    NetworkParams = namedtuple("NetworkParams", ["runtime"])

    def __init__(self, networks, sequence, device, frozen, rearrange_wrappers=True):
        assert len(networks) == 2 # Currently tested only for a sequence of 2 networks
        first_net = networks[sequence[0]]
        self.last_net = networks[sequence[1]]
        super().__init__(networks, sequence, frozen)
        self.model = self.last_net.model

        if rearrange_wrappers:
            # Handle wrappers
            self.wrappers = self.last_net.wrappers
            self.last_net.wrappers = self.initialize_wrappers("", device)

            # Handle params
            self.network_params = self.NetworkParams({"wrappers": self.last_net.network_params.runtime["wrappers"],
                                                      "data": first_net.network_params.runtime["data"]})
        else:
            # Handle params
            self.network_params = self.NetworkParams({"wrappers": "", "data": first_net.network_params.runtime["data"]})

        assert first_net.meta["out_channels"] == self.last_net.meta["in_channels"]
        self.meta = {"in_channels": first_net.meta["in_channels"], "out_channels": self.last_net.meta["out_channels"]}

    def __getattr__(self, name):
        return getattr(self.last_net, name)

    def forward(self, image):
        return self.wrappers[self.stage](image, self.forward_batch, outputmodel=self.model)

    def forward_batch(self, images):
        if images is None:
            return None
        if isinstance(images, list):
            return [self._forward_all(x) for x in images]
        return self._forward_all(images)

    def _forward_all(self, image):
        for net in self.network_order:
            image = self.networks[net](image)
        return image

    @classmethod
    def initialize(cls, params, device):
        sequence = params.pop("sequence").split(",")
        rearrange_wrappers = params.pop("rearrange_wrappers") if "rearrange_wrappers" in params else True
        networks = cls._initialize_subnetworks(params, device)
        return cls(networks, sequence, device=device, frozen=False, rearrange_wrappers=rearrange_wrappers)

    def overlay_params(self, new_params, device):
        if not new_params:
            return self

        sequence = new_params.pop("sequence").split(",") if "sequence" in new_params else self.network_order
        rearrange_wrappers = new_params.pop("rearrange_wrappers") if "rearrange_wrappers" in new_params else True
        networks = self._overlay_subnetworks(new_params, device)
        return self.__class__(networks, sequence, device=device, frozen=True, rearrange_wrappers=rearrange_wrappers)

    #
    # Load and save
    #

    def state_dict(self):
        state = super().state_dict()
        state["net"]["sequence"] = state["net"].pop("network_order")
        return state

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        checkpoint = state_dict.pop("net")
        assert checkpoint["type"] == cls.__name__
        assert checkpoint.keys() == {"type", "frozen", "sequence", "network_hierarchy"}, checkpoint.keys()
        assert set(checkpoint["sequence"]) == checkpoint['network_hierarchy'].keys()

        # Propagate runtime
        runtime_propagated = {net: None for net in checkpoint["sequence"]}
        if runtime and "wrappers" in runtime:
            runtime_propagated[checkpoint['sequence'][-1]] = {"wrappers": runtime.pop("wrappers")}
        if runtime and "data" in runtime:
            runtime_propagated[checkpoint['sequence'][0]] = {"data": runtime.pop("data")}
        assert not runtime, runtime

        # Handle this net's parameters
        if params is not None:
            params_sequence = params.pop("sequence").split(",")
            assert checkpoint["sequence"] == params_sequence, \
                    "%s != %s" % (str(checkpoint["sequence"]), str(params_sequence))

        networks = cls._initialize_subnetworks_from_state(checkpoint["network_hierarchy"], state_dict, device, params, runtime_propagated)
        return cls(networks, checkpoint["sequence"], device=device, frozen=checkpoint["frozen"])

    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.network_params._asdict().items())
        nice_wrappers = "\n" + "".join("    %s: %s,\n" % (x, indent(str(y))) for x, y in self.wrappers.items())
        nice_networks = ""
        for net in self.network_order:
            nice_networks += f"        {net}: {indent(str(self.networks[net]), 2)}\n"

        return \
f"""{self.__class__.__name__} (
    sequence: {self.network_order}
    networks: {{
{nice_networks}
    }}
    meta: {self.meta}
    network_params: {{{indent(nice_params)}}}
    wrappers: {{{indent(nice_wrappers)}}}
)"""


class CirSequentialNetwork(SequentialNetwork):

    # Do not call forward on each individual image in images
    def forward_batch(self, images):
        if images is None:
            return None
        return self._forward_all(images)


class MultiheadNetwork(MultiNetwork):
    """
    Expects base, split and multiple head networks (in this order in network_order, names can be
    arbitrary).

    Architecture: base (param - e.g. VGG) -> split (param) -> heads (e.g. l2 for descriptor / classifier)
    """

    NetworkParams = namedtuple("NetworkParams", ["runtime", "parameter_groups"])

    def __init__(self, networks, network_order, network_params, device, frozen):
        super().__init__(networks, network_order, frozen)
        assert networks.keys() == set(network_order)
        self.network_params = network_params
        self._base, self._split, *self._head_order = network_order

        # Handle input and output net
        self.default_output = network_params.runtime["default_output"]
        assert self.default_output in self.network_order and self.default_output != self._split, self.default_output

        # Handle params
        self.wrappers = self.initialize_wrappers(network_params.runtime.get("wrappers", ""), device)
        if "data" not in self.network_params.runtime:
            self.network_params.runtime["data"] = networks[self._base].network_params.runtime["data"]

        out_channels = networks[self._split].meta["out_channels"]
        assert isinstance(out_channels, tuple) and len(out_channels) == len(self._head_order)
        if not networks[self._base].meta:
            in_channels = networks[self._split].meta["in_channels"]
            networks[self._base].set_meta({"in_channels": in_channels, "out_channels": in_channels})
        assert networks[self._base].meta["out_channels"] == networks[self._split].meta["in_channels"]
        for i, head in enumerate(self._head_order):
            out_channels = networks[self._split].meta["out_channels"][i]
            if not self.networks[head].meta:
                self.networks[head].set_meta({"in_channels": out_channels, "out_channels": out_channels})
            assert out_channels == self.networks[head].meta["in_channels"]

        self.set_meta({"in_channels": networks[self._base].meta["in_channels"], "out_channels": networks[self.default_output].meta["out_channels"]})

    def __call__(self, image):
        return self.wrappers[self.stage](image, self._forward_inference, outputmodel=self.networks[self.default_output].model)

    def _forward_inference(self, image):
        return self.forward_batch(image, single_output=self.default_output)

    def forward(self, image):
        return self.wrappers[self.stage](image, self.forward_batch, outputmodel=self.networks[self.default_output].model)

    def forward_batch(self, images, *, single_output=None):
        if images is None:
            return None
        if isinstance(images, list):
            return [self._forward_all(x, single_output) for x in images]
        return self._forward_all(images, single_output)

    def _forward_all(self, image, single_output):
        """
        Report all intermediate results, propagating the image through the chain of networks.
        If single_output specified, report only results of this network and stop execution right after
        that, skipping all network not necessary for this output.
        """
        results = {}
        # Base
        base = self.networks[self._base](image)
        results[self._base] = base
        # Return only desired output
        if single_output == self._base:
            return base

        # Split
        pieces = self.networks[self._split](base)
        assert len(pieces) == len(self._head_order)

        # Execute only desired head
        if single_output in self._head_order:
            inp = pieces[self._head_order.index(single_output)]
            return self.networks[single_output](inp)
        assert single_output is None, single_output

        # Execute and report all heads
        for inp, head in zip(pieces, self._head_order):
            results[head] = self.networks[head](inp)
        return results

    @classmethod
    def initialize(cls, params, device):
        network_order = params.pop("network_order").split(",")
        network_params = cls.NetworkParams(params.pop("runtime"), params.pop("parameter_groups"))
        networks = cls._initialize_subnetworks(params, device)
        return cls(networks, network_order, network_params, device=device, frozen=False)

    def overlay_params(self, new_params, device):
        if not new_params:
            return self

        network_params = self.NetworkParams(new_params.pop("runtime"), self.network_params.parameter_groups)
        networks = self._overlay_subnetworks(new_params, device)
        return self.__class__(networks, self.network_order, network_params, device=device, frozen=True)

    def parameters(self, optimizer_opts, net=None):
        return super()._parameters_with_groups(optimizer_opts, net, parameter_groups=self.network_params.parameter_groups)


    #
    # Load and save
    #

    def state_dict(self):
        state = super().state_dict()
        state["net"]["network_params"] = self.network_params._asdict()
        return state

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        checkpoint = state_dict.pop("net")
        assert checkpoint["type"] == cls.__name__
        assert checkpoint.keys() == {"type", "frozen", "network_params", "network_order", "network_hierarchy"}, checkpoint.keys()
        assert set(checkpoint["network_order"]) == checkpoint['network_hierarchy'].keys()

        network_params = cls.NetworkParams(**checkpoint["network_params"])
        network_params.runtime.update(runtime or {})

        networks = cls._initialize_subnetworks_from_state(checkpoint["network_hierarchy"], state_dict, device, params, None)
        return cls(networks, checkpoint["network_order"], network_params, device=device, frozen=checkpoint["frozen"])


    #
    # Utils
    #

    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.network_params._asdict().items())
        nice_wrappers = "\n" + "".join("    %s: %s,\n" % (x, indent(str(y))) for x, y in self.wrappers.items())
        nice_networks = ""
        for net in self.network_order:
            nice_networks += f"        {net}: {indent(str(self.networks[net]), 2)}\n"

        return \
f"""{self.__class__.__name__} (
    network_order: {self.network_order}
    networks: {{
{nice_networks}
    }}
    meta: {self.meta}
    network_params: {{{indent(nice_params)}}}
    wrappers: {{{indent(nice_wrappers)}}}
)"""


# Initialization

NETWORKS = {
    "SingleNetwork": SingleNetwork,
    "SingleNetworkLink": SingleNetworkLink,
    "CirNetwork": CirNetwork,
    "GlobalLocalNetwork": GlobalLocalNetwork,

    "MultiNetwork": MultiNetwork,
    "NetworkSet": NetworkSet,
    "SequentialNetwork": SequentialNetwork,
    "CirSequentialNetwork": CirSequentialNetwork,
    "MultiheadNetwork": MultiheadNetwork,
}

def initialize_network(params, device, state=None, runtime=None):
    network_cls = NETWORKS[params.pop("type") if params else state["net"]["type"]]

    if state:
        return network_cls.initialize_from_state(state, device, params, runtime)

    return network_cls.initialize(params, device)
