import re
import random
import json
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from daan.data.fs_driver import fs_driver
from daan.core.path_resolver import resolve_path
from ...tools import utils, tensors
from .transform import functional as tfunc


class Compose(object):
    """Take multiple wrappers and apply them sequentially. In forward direction for preprocess,
        in backward direction for postprocess."""

    def __init__(self, wrappers, device):
        """Store a list of wrappers"""
        self.wrappers = wrappers
        self.device = device

    def __call__(self, tensor, inference, outputmodel=None, tensor_params=None):
        """Apply wrappers sequentially for preprocess, evaluate model and apply wrappers
            sequentially for postprocess in reversed order. Return tensor."""
        tensor_params = {} if tensor_params is None else tensor_params
        if not self.wrappers:
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(self.device)
            return inference(tensor, **tensor_params)

        if outputmodel is None:
            outputmodel = inference

        metadata = []
        for wrapper in self.wrappers:
            tensor, meta = wrapper.preprocess(tensor, outputmodel)
            metadata.append(meta)

        tensor = inference(tensors.to_device(tensor, self.device), **tensor_params)

        for wrapper, meta in reversed(list(zip(self.wrappers, metadata))):
            tensor = wrapper.postprocess(tensor, outputmodel, meta)
        return tensor

    def __repr__(self):
        nice_wrappers = "\n" + "".join("    %s\n" % x for x in self.wrappers) if self.wrappers else ""
        return f"""{self.__class__.__name__}([{nice_wrappers}])"""


class Wrapper(object):
    """Wrap network input and output with custom functions to support various inference patterns
        such as tiling. This should be used only during inference"""

    def __init__(self, device):
        pass

    def preprocess(self, tensor, _outputmodel):
        """Return unmodified input tensor and None for metadata. Applied on network input."""
        return tensor, None

    def postprocess(self, tensor, _outputmodel, _metadata):
        """Return unmodified input tensor. Applied on network output."""
        return tensor


class ReflectPadMakeDivisible(Wrapper):
    """Pad tensor so that its spatial dimension is divisible by a specified number. Pad it using
        reflection around the boundary."""

    def __init__(self, divisible_by, device):
        """Store divisible_by int"""
        super().__init__(device)
        self.divisible_by = int(divisible_by) if isinstance(divisible_by, str) else divisible_by

    def preprocess(self, tensor, outputmodel):
        """Return padded tensor and used padding"""
        if isinstance(tensor, list):
            prep = list(zip(*[self.preprocess(t, outputmodel) for t in tensor]))
            return list(prep[0]), list(prep[1])
        size = np.array(tensor.size())[2:]
        padx, pady = (np.ceil(size / self.divisible_by)*self.divisible_by - size) / 2
        padding = (int(np.floor(pady)), int(np.ceil(pady)), int(np.floor(padx)), int(np.ceil(padx)))
        return F.pad(tensor, padding, 'replicate'), padding

    def postprocess(self, tensor, outputmodel, padding):
        """Return cropped tensor based on given padding"""
        if isinstance(tensor, list):
            return [self.postprocess(t, outputmodel, p) for (t, p) in zip(tensor, padding)]
        return tensor[:,:,padding[2]:-padding[3] or None,padding[0]:-padding[1] or None]

    def __repr__(self):
        return f"{self.__class__.__name__} (divisible_by={self.divisible_by})"


class RandomPassThrough(Wrapper):
    """Lets the input to be processed with wrapped network by given probability.
        Otherwise the input skips the network."""

    def __init__(self, probability_through, device):
        super().__init__(device)
        self.probability = float(probability_through)

    def preprocess(self, tensor, outputmodel):
        if isinstance(tensor, list):
            out = tuple(zip(*[self.preprocess(t, outputmodel) for t in tensor]))
            return list(out[0]), list(out[1])
        return (tensor, None) if random.random() < self.probability else (None, tensor)

    def postprocess(self, tensor, outputmodel, tensor_skipping):
        if isinstance(tensor, list):
            return [self.postprocess(t, outputmodel, s) for (t, s) in zip(tensor, tensor_skipping)]
        return tensor if tensor_skipping is None else tensors.as_tensor(tensor_skipping)

    def __repr__(self):
        return f"{self.__class__.__name__}(probability={self.probability})"


class CirRatioPassThrough(RandomPassThrough):

    def __init__(self, ratio_through, image_label, *, device):
        super().__init__(ratio_through, device)
        self.image_label = re.compile(image_label)

    def preprocess(self, tensor, outputmodel):
        if isinstance(tensor, list):
            acc = [self.preprocess(x, outputmodel) for x in tensor]
            return tuple(list(x) for x in zip(*acc))

        image_label = tensor.metadata['image_label'] # image_label must be present - sanity check
        if isinstance(image_label, list) and len(image_label) == 1:
            image_label = image_label[0]
        if self.image_label.match(image_label) and self._passthrough(tensor.metadata['name']):
            return tensor, None
        return None, tensor

    def _passthrough(self, name):
        if isinstance(name, list):
            name, = name
        digits = 4 # 16 ** digits = precision
        rand = int(hashlib.md5(name.encode("utf8")).hexdigest()[-digits:], 16) / (16**digits)
        return rand < self.probability

    def __repr__(self):
        return f"{self.__class__.__name__}(probability={self.probability}, train_label={self.image_label})"


class MeanStdPost(Wrapper):
    """Adapts mean and std of the input tensor distribution into the given output."""

    def __init__(self, input_meanstd, output_meanstd, device):
        super().__init__(device)
        input_meanstd, output_meanstd = json.loads(input_meanstd), json.loads(output_meanstd)
        self.input_meanstd = [self.mean2tensor(x, device) for x in input_meanstd]
        self.output_meanstd = [self.mean2tensor(x, device) for x in output_meanstd]
        if any(x == 0 for x in input_meanstd[1]) or any(x == 0 for x in output_meanstd[1]):
            raise ValueError("Some std element is zero, leading to zero division.")

    @staticmethod
    def mean2tensor(mean, device):
        mean = torch.as_tensor(mean, device=device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        return mean

    def postprocess(self, tensor, outputmodel, meta):
        if isinstance(tensor, list):
            return [self.postprocess(x, outputmodel, meta) for x in tensor]
        return self._adapt(tensor)

    def _adapt(self, tensor):
        tensor = tensor.mul(self.input_meanstd[1]).add(self.input_meanstd[0])  # unnormalize
        tensor = tensor.sub(self.output_meanstd[0]).div(self.output_meanstd[1])  # normalize
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(input_meanstd={self.input_meanstd}," \
            f"output_meanstd={self.output_meanstd})"


class MeanStdPre(MeanStdPost):
    """Adapts mean and std of the input tensor distribution into the given output."""

    def __init__(self, input_meanstd, output_meanstd, device):
        super().__init__(input_meanstd, output_meanstd, device)

    def preprocess(self, tensor, _outputmodel):
        if isinstance(tensor, list):
            return [self.preprocess(x, _outputmodel) for x in tensor]
        return self._adapt(tensor), None

    def postprocess(self, tensor, outputmodel, meta):
        return tensor


class CirMultiscaleAggregation(Wrapper):
    """Downscale each image to defined scales and aggregate resulting descriptors."""

    def __init__(self, scales, device):
        """Parse and store scales"""
        super().__init__(device)
        if isinstance(scales, str):
            scales = {"True": True, "False": False, "ms": True, "ss": False,
                      "sms5": [1, 1./np.sqrt(2), np.sqrt(2), 1./2, 2],
                      "sms": [1, 1./np.sqrt(2), np.sqrt(2)]}[scales]
        if isinstance(scales, bool):
            scales = [1, 1./np.sqrt(2), 1./2] if scales else [1]
        self.scales = scales

    @staticmethod
    def wrap_metadata(tensor, func):
        metadata = None
        if hasattr(tensor, "metadata") and hasattr(tensor, "tensor"):
            tensor, metadata = tensor.tensor, tensor.metadata
        tensor = func(tensor)
        if metadata is None:
            return tensor
        return tensors.as_metadata_tensor(tensor, metadata)

    def preprocess(self, tensor, _outputmodel):
        if len(self.scales) == 1:
            return tensor if isinstance(tensor, list) else [tensor], isinstance(tensor, list)

        interpolate = lambda x, y: F.interpolate(x, scale_factor=y, mode='bilinear', align_corners=False)
        if isinstance(tensor, list):
            acc = []
            for single in tensor:
                for scale in self.scales:
                    acc.append(self.wrap_metadata(single, lambda x: interpolate(x, scale)))
            return acc, True

        return [self.wrap_metadata(tensor, lambda x: interpolate(x, scale)) for scale in self.scales], False

    @staticmethod
    def aggregate_tensor(tensor, nscales, outputdim, msp):
        assert len(tensor) == nscales, "%s != %s" % (len(tensor), nscales)
        v = torch.zeros(outputdim, dtype=tensor[0].dtype, device=tensor[0].device)
        for subtensor in tensor:
            v += subtensor.pow(msp).squeeze()

        v = (v / nscales).pow(1./msp)
        v /= v.norm()

        return v

    def postprocess(self, tensor, outputmodel, waslist):
        msp = 1
        if len(self.scales) > 1 and outputmodel.meta.get("pooling", None) == 'gem' \
                and not outputmodel.meta['regional'] and not outputmodel.meta['whitening']:
            msp = outputmodel.pool.p.item()

        if not waslist:
            return self.aggregate_tensor(tensor, len(self.scales), outputmodel.meta['out_channels'], msp)

        assert len(tensor) % len(self.scales) == 0, "%s %% %s != 0" % (len(tensor), len(self.scales))
        acc = []
        for i in range(0, len(tensor), len(self.scales)):
            acc.append(self.aggregate_tensor(tensor[i:i+len(self.scales)], len(self.scales), outputmodel.meta['out_channels'], msp))
        return acc

    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales})"


class FakeBatch(Wrapper):
    """Mimic batch behaviour by accumulating the result across multiple images"""

    def postprocess(self, tensor, outputmodel, _meta):
        if not isinstance(tensor, list) or not isinstance(tensor[0], torch.Tensor):
            return tensor

        output = torch.zeros(outputmodel.meta['out_channels'], len(tensor), device=tensor[0].device)
        for j, vec in enumerate(tensor):
            output[:, j] = vec.squeeze()
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CirFakeTupleBatch(FakeBatch):
    """Mimic batch behaviour by accumulating the result across multiple images in multiple tuples"""

    @classmethod
    def unsqueeze(cls, tensor):
        if isinstance(tensor, list):
            return [cls.unsqueeze(x) for x in tensor]
        elif len(tensor.shape) == 3:
            return tensor.unsqueeze_(0)
        elif len(tensor.shape) == 4:
            return tensor
        raise ValueError("Unsupported tensor dimensionality %s" % len(tensor.shape))

    def preprocess(self, tensor, _outputmodel):
        """Flatten the 2d list"""
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            return tensor, False

        acc = []
        meta = len(tensor[0])
        for tpl in tensor:
            assert meta == len(tpl)
            acc += tpl
        return acc, meta


class CirtorchWhiten(Wrapper):
    """Whiten vectors with possible dimensionality reduction"""

    def __init__(self, whitening, dimensions, device):
        """Load whitening by its whitening path and store dimensions for reduction (empty for
            disable)"""
        super().__init__(device)
        whitening = fs_driver(resolve_path(whitening)).load()
        self.P = torch.tensor(whitening['P'], dtype=torch.float32, device=device)
        self.m = torch.tensor(whitening['m'], dtype=torch.float32, device=device)
        self.dimensions = dimensions or self.P.shape[0]

    def postprocess(self, tensor, _outputmodel, _meta):
        X = self.P[:self.dimensions, :].mm(tensor.unsqueeze_(1).sub_(self.m))
        return X.div_(torch.norm(X, p=2, dim=0, keepdim=True) + 1e-6).squeeze()


class ClahePost(Wrapper):

    def __init__(self, meanstd, clip_limit=4, grid_size=8, colorspace="lab", *, device):
        """Apply non-differentiable cv2 CLAHE as a post-processing"""
        super().__init__(device)
        self.meanstd = [MeanStdPost.mean2tensor(x, device) for x in json.loads(meanstd)]
        self.clahe = tfunc.ImageClahe(clip_limit=float(clip_limit), grid_size=int(grid_size),
                                      colorspace=colorspace)

    def postprocess(self, tensor, outputmodel, meta):
        if tensor is None:
            return tensor
        if isinstance(tensor, list):
            return [self.postprocess(x, outputmodel, meta) for x in tensor]
        if len(tensor.shape) == 4:
            return torch.stack([self.postprocess(x, outputmodel, meta) for x in tensor])
        if len(tensor.shape) == 3:
            tensor = tensor.detach().mul(self.meanstd[1]).add(self.meanstd[0]) # unnormalize
            img = tensor.cpu().numpy().transpose((1, 2, 0)) # to numpy
            img = self.clahe.apply(img) # transform
            tensor = torch.from_numpy(img.transpose((2, 0, 1))).to(tensor.device) # from numpy
            tensor = tensor.sub(self.meanstd[0]).div(self.meanstd[1]) # normalize
            return tensor
        raise ValueError(f"Unsupported tensor dims: {len(tensor.shape)}")


class RgbToBgrPre(Wrapper):

    def __init__(self, device):
        """Converts torch tensor in RGB to BGR colorspace."""
        super().__init__(device)

    def preprocess(self, tensor, _outputmodel):
        if isinstance(tensor, list):
            return [self.preprocess(x, _outputmodel) for x in tensor], None
        if len(tensor.shape) == 4:
            return tensor[:, [2, 1, 0], ...], None
        if len(tensor.shape) == 3:
            return tensor[[2, 1, 0], ...], None
        raise ValueError(f"Unsupported tensor dims: {tensor.shape}")


WRAPPERS_LABELS = {
    "reflectpad_divisible": ReflectPadMakeDivisible,
    "random_pass_through": RandomPassThrough,
    "cir_ratio_pass_through": CirRatioPassThrough,
    "meanstd_post": MeanStdPost,
    "meanstd_pre": MeanStdPre,
    "cirmultiscale": CirMultiscaleAggregation,
    "fakebatch": FakeBatch,
    "cirfaketuplebatch": CirFakeTupleBatch,
    "cirwhiten": CirtorchWhiten,
    "clahepost": ClahePost,
    "rgb2bgr_pre": RgbToBgrPre,
}


# Init function

def initialize_wrappers(net_wrappers, device):
    if net_wrappers is None:
        wraps = []
    elif isinstance(net_wrappers, str):
        wraps = []
        splitted = utils.splitp(net_wrappers, ",", check_valid_pairs=True)
        for wrap in [x.strip() for x in splitted if x]:
            wname, *args = wrap.split(":")
            wraps.append(WRAPPERS_LABELS[wname](*args, device=device))
    else:
        wraps = [WRAPPERS_LABELS[x.split("_", 1)[1]](**net_wrappers[x], device=device) \
                    for x in sorted(net_wrappers)]
    return Compose(wraps, device)
