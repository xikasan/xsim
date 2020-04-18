# coding: utf-8

import numpy as np
from ..rungekutta import no_time_rungekutta
from .base import BaseModel


class Filter1st(BaseModel):

    def __init__(self, dt, tau, gain=1, init_val=None, dtype=np.float32):
        self.dt = dt
        self.tau = tau
        self.gain = gain

        self.x = init_val
        self.dx = 0.0

        self.init_val = init_val if init_val is not None else 0.0
        self.dtype = dtype

    def __call__(self, action):
        def fn(x):
            return (self.gain * action - x) / self.tau
        dx = no_time_rungekutta(fn, self.dt, self.x)
        self.x += dx * self.dt
        return self.get_state()

    def reset(self, init_val=None):
        init_val = init_val if init_val is not None else self.init_val
        self.x = init_val
        self.dx = 0
        return self.get_state()

    def get_state(self):
        x = self.dtype(self.x)
        return x

    def get_full_state(self):
        xs = np.array([self.x, self.dx], dtype=self.dtype)
        return xs
