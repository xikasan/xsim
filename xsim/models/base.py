# coding: utf-8

import numpy as np
from cached_property import cached_property


class Space:

    def __init__(self, high, low, dtype=np.float32):
        high = np.asanyarray(high, dtype=dtype)
        low  = np.asanyarray(low,  dtype=dtype)
        assert len(high.shape) < 2 and len(low.shape) < 2,\
            "dimension of high and low must be < 2, but high:{} and low:{} is given".format(
                len(high.shape), len(low.shape)
            )
        assert high.shape == low.shape,\
            "dimension of high and low must be same, but high:{} and low:{} are given".format(
                high.shape, low.shape
            )
        self._is_scalar = len(high.shape) == 0
        self.high  = high
        self.low   = low
        self.dtype = dtype

    @cached_property
    def size(self):
        if self._is_scalar:
            return 1
        return self.high.shape[0]


class BaseModel:

    def __init__(self, dt, dtype=np.float32, name="BaseModel"):
        self.dt = dt
        self.dtype = dtype
        self.name = name

        self.action_space = None
        self.state_space = None

    def __call__(self, action):
        raise NotImplementedError()


if __name__ == '__main__':
    space = Space(2, -2)
    print(space.high, space.low)
    print(len(space.high.shape))
    print(space._is_scalar)
