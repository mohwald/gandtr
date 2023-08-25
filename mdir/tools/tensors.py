"""
Helper functions to simplify working with tensors
"""
from typing import NamedTuple, Any
import torch


def to_device(tensor, device):
    """Casts tensors nested in lists and dicts to a given device while keeping their structure"""
    if hasattr(tensor, "to"):
        return tensor.to(device)
    if isinstance(tensor, list):
        return [to_device(x, device) for x in tensor]
    if isinstance(tensor, tuple):
        return tuple(to_device(x, device) for x in tensor)
    if isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    if tensor is None:
        return None
    return tensor.to(device)


def detach(tensor):
    """Detach tensors nested in lists and dicts while keeping their structure"""
    if hasattr(tensor, "detach"):
        return tensor.detach()
    if isinstance(tensor, list):
        return [detach(x) for x in tensor]
    if isinstance(tensor, tuple):
        return tuple(detach(x) for x in tensor)
    if isinstance(tensor, dict):
        return {k: detach(v) for k, v in tensor.items()}
    if tensor is None:
        return None
    return tensor.detach()


class MetadataTensor(NamedTuple):
    """Class that implements both NamedTuple (for batch default_collate) and torch.Tensor extension
    (https://pytorch.org/docs/stable/notes/extending.html)"""

    tensor: Any
    metadata: Any

    def __repr__(self):
        return f"Data:\n{self.tensor}\nMetadata:\n{self.metadata}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        metadata, = (a.metadata for a in args if isinstance(a, cls))
        args = [a.tensor if isinstance(a, cls) else a for a in args]
        return MetadataTensor(func(*args, **(kwargs or {})), metadata)

    def __getattr__(self, name):
        attr = getattr(self.tensor, name)
        if name not in ["to", "unsqueeze_"]:
            # Returning tensor's attr by default
            return attr

        def func(*args, **kwargs):
            """Wrapper around a method that need's to be propagated propagated, but without changing
            the resulting datatype"""
            return self.__class__(attr(*args, **kwargs), self.metadata)
        return func


def as_metadata_tensor(tensor, metadata):
    if isinstance(tensor, MetadataTensor):
        assert not set(tensor.metadata.keys()).intersect(metadata.keys())
        tensor.metadata.update(dict(metadata))
        return tensor
    return MetadataTensor(torch.as_tensor(tensor), dict(metadata))

def as_tensor(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, MetadataTensor):
        return tensor.tensor
    if isinstance(tensor, list):
        return [as_tensor(x) for x in tensor]
    if isinstance(tensor, tuple):
        return tuple(as_tensor(x) for x in tensor)
    if isinstance(tensor, dict):
        return {k: as_tensor(v) for k, v in tensor.items()}
    return torch.as_tensor(tensor)
