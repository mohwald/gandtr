"""
Wrappers around loss values simplifying the work with multi-number loss values
"""

import operator
import torch


class Zero:
    """
    Represents zero where the zero object is replaced with a second operand after e.g. addition.
    Enables to change the class in the first addition which makes code more readable when
    e.g. statistics are to be taken and the value type is unknown.
    """

    def __add__(self, obj):
        return obj
    def __sub__(self, obj):
        return -obj
    def __mul__(self, obj):
        return self
    def __truediv__(self, obj):
        return self
    def __str__(self):
        return f"{self.__class__.__name__}()"

ZERO = Zero()


class MultiValue:
    """
    Represents a loss value that has multiple numbers associated with
    """


class TotalWithIntermediate(MultiValue):
    """
    Represents a total value (loss) with possibly multiple intermediate results given as kwargs
    to the constructor.
    """

    def __init__(self, total, **partial):
        self.total = total
        self.partial = self._flatten_partial(partial)

    @classmethod
    def from_partial(cls, **partial):
        """Compute total from partial and return TotalWithIntermediate instance"""
        partial = cls._flatten_partial(partial)
        total = ZERO
        for partial in partial.values():
            total = total + partial # Cannot use += as it changes the tensor on the left
        return TotalWithIntermediate(total, **partial)

    @classmethod
    def _flatten_partial(cls, partial):
        partial_flat = {}
        for key, value in partial.items():
            if isinstance(value, cls):
                # Flatten nested TotalWithIntermediate
                for subkey, subvalue in value.partial.items():
                    partial_flat["%s.%s" % (key, subkey)] = subvalue
                value = value.total
            partial_flat[key] = value
        return partial_flat

    def backward(self, **kwargs):
        """When type of values is torch.Tensor"""
        return self.total.backward(**kwargs)

    def item(self):
        """When type of values is torch.Tensor"""
        return self.__class__(self.total.item(), **{x: y.item() for x, y in self.partial.items()})

    def __iter__(self):
        """Enables conversion to dict"""
        return iter([("total", self.total)] + list(self.partial.items()))

    @classmethod
    def _binary_operator(cls, obj1, obj2, oper, this=True, scalar=True):
        assert isinstance(obj1, cls)
        # When used with Tensor
        if isinstance(obj2, torch.Tensor):
            return oper(obj1.total, obj2)
        # When used with TotalWithIntermediate
        if this and isinstance(obj2, cls):
            assert obj1.partial.keys() == obj2.partial.keys()
            partial = {x: oper(obj1.partial[x], obj2.partial[x]) for x in obj1.partial}
            return cls(oper(obj1.total, obj2.total), **partial)
        # When used with scalar
        if scalar and isinstance(obj2, (int, float)):
            partial = {x: oper(obj1.partial[x], obj2) for x in obj1.partial}
            return cls(oper(obj1.total, obj2), **partial)

        raise TypeError(f"Unsupported second operand type {type(obj2)} in operation {oper}")

    def __add__(self, obj):
        return self._binary_operator(self, obj, operator.add, scalar=False)

    def __sub__(self, obj):
        return self._binary_operator(self, obj, operator.sub, scalar=False)

    def __mul__(self, obj):
        return self._binary_operator(self, obj, operator.mul, this=False)

    def __truediv__(self, obj):
        return self._binary_operator(self, obj, operator.truediv, this=False)

    def __float__(self):
        return float(self.total)

    def __str__(self):
        partial = "".join(f", {x}={y}" for x, y in self.partial.items())
        return f"{self.__class__.__name__}({self.total}{partial})"

    __radd__ = __add__
    __rmul__ = __mul__
