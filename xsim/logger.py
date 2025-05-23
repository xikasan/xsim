# coding: utf-8

import numpy as np
import pandas as pd
from .batch import Batch


class Logger:

    def __init__(self, size=1000, dtype=np.float32):
        self.dtype = dtype
        self.size = size
        self._buf = {}

        self._counter = 0
        self._current_data = {}

    def __len__(self):
        return self._counter

    def store(self, **kwargs):
        for key, val in kwargs.items():
            self._current_data[key] = np.squeeze(val)
        return self

    def flush(self):
        for key, val in self._current_data.items():
            self._add_key_if_not_exist(key, val)
            self._add_capacity_if_necessary(key)
            self._buf[key][self._counter, :] = val
        self._counter += 1
        self._current_data = {}

    def latest(self, size):
        first = (self._counter - size) if self._counter > size else 0
        last  = self._counter
        batch_data = {key: val[first:last] for key, val in self._buf.items()}
        batch = Batch.make(batch_data)
        return batch

    def buffer(self):
        return {key: val[0:self._counter] for key, val in self._buf.items()}

    def _add_capacity_if_necessary(self, key):
        # check capacity
        if self._counter < self._buf[key].shape[0]:
            return
        # add capacity
        new_buffer = np.zeros((self.size, self._buf[key].shape[1]), dtype=self.dtype)
        self._buf[key] = np.concatenate([self._buf[key], new_buffer], axis=0)

    def _add_key_if_not_exist(self, key, val):
        # existing check
        if key in self._buf.keys():
            return

        # find length
        size = self._buf[list(self._buf.keys())[0]].shape[0] \
            if len(self._buf.keys()) > 0 \
            else self.size
        # find width
        shape = val.shape
        shape = 1 if len(shape) == 0 else shape[0]

        # add new buffer
        self._buf[key] = np.zeros((size, shape), dtype=self.dtype)


class Retriever:

    def __init__(self, source):
        assert isinstance(source, (dict, Logger)), \
            "type of data should be dict or xsim.Logger, but {} is given".format(type(source))
        self._source = source if isinstance(source, dict) else source.buffer()
        for key, value in self._source.items():
            self.__setattr__(key, RetrieverData(value))

    def __call__(self, key, idx=None):
        temp = self._source[key]
        if idx is not None:
            return np.squeeze(temp)
        return np.squeeze(temp[:, idx])

    def to_dataframe(self):
        vdict = dict()
        for key, value in self._source.items():
            shape = value.shape
            if shape[1] == 1:
                vdict[key] = value.squeeze()
            else:
                for i in range(shape[1]):
                    vdict[key+f"_{i}"] = value[:, i].squeeze()
        return pd.DataFrame(vdict)


class RetrieverData:

    def __init__(self, data):
        self.data = np.squeeze(data)

    def __call__(self, idx=None, fn=None):
        d = self.__retrieve(idx=idx)
        if fn is None:
            return d
        return fn(d)

    def __retrieve(self, idx):
        if idx is None:
            return np.squeeze(self.data)
        return np.squeeze(self.data[:, idx])
