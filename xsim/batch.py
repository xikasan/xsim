# coding: utf-8

import numpy as np


class Batch:

    def __init__(self, size, dtype=np.float32):
        self.size = size
        self.dtype = dtype

    def append(self, key, value):
        self.__setattr__(key, value)

    @staticmethod
    def make(data, squeeze=False):
        size = data[list(data.keys())[0]].shape[0]
        batch = Batch(size)
        for key, val in data.items():
            if squeeze:
                val = np.squeeze(val)
            batch.append(key, val)
        return batch
