# coding: utf-8

import numpy as np


def bounded_normal(shape=1):
    num = np.prod(shape)
    rs = np.array([_bounded_normal() for _ in range(num)])
    rs = rs.reshape(shape)
    return rs


def _bounded_normal():
    while True:
        r = np.random.normal(0, 0.2)
        if -1 <= r <= 1:
            return r
