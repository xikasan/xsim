# coding: utf-8

import numpy as np


class Batch:

    def __init__(self, dtype=np.float32):
        self.size = None
        self.dtype = dtype

    def append(self, key, value):
        if self.size is None:
            self.size = value.shape[0]
        self.__setattr__(key, value)

    @staticmethod
    def make(data):
        size = data[list(data.keys())[0]].shape[0]
        batch = Batch(size)
        for key, val in data.items():
            val = np.squeeze(val)
            batch.append(key, val)
        return batch
