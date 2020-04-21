# coding: utf-8

import numpy as np
from .logger import Logger


class Batch:

    def __init__(self, dtype=np.float32):
        self.size = None
        self.dtype = dtype

    def append(self, key, value):
        if self.size is None:
            self.size = value.shape[0]
        self.__setattr__(key, value)

    @staticmethod
    def make_batch(data):
        size = data[list(data.keys())[0]].shape[0]
        batch = Batch(size)
        print(size)


class Retriever:

    def __init__(self, source):
        assert isinstance(source, [dict, Logger]), \
            "type of data should be dict or xsim.Logger, but {} is given".format(type(source))
        self._source = source if isinstance(source, dict) else source.buffer()

    def __call__(self, key, idx=None):
        temp = self._source[key]
        if idx is not None:
            return np.squeeze(temp)
        return np.squeeze(temp[:, idx])
