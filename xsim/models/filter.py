# coding: utf-8

import numpy as np
from ..rungekutta import no_time_rungekutta
from .base import BaseModel


class Filter1st(BaseModel):

    def __init__(self, dt, tau, gain=1, init_val=None, dtype=np.float32, name="Filter1st"):
        super().__init__(dt, dtype=dtype, name=1)

        # parameters
        self.tau = tau
        self.gain = gain
        self.init_val = init_val if init_val is not None else 0.0

        # states
        self.x = init_val
        self.dx = 0.0

    def __call__(self, action):
        def fn(x):
            return (self.gain * action - x) / self.tau
        self.dx = no_time_rungekutta(fn, self.dt, self.x)
        self.x += self.dx * self.dt
        return self.get_state()

    def reset(self, init_val=None):
        init_val = init_val if init_val is not None else self.init_val
        self.x = init_val
        self.dx = 0.0
        return self.get_state()

    def get_state(self):
        x = self.dtype(self.x)
        return x

    def get_full_state(self):
        xs = np.array([self.x, self.dx], dtype=self.dtype)
        return xs


class Filter2nd(BaseModel):

    def __init__(self, dt, tau, gain=1, init_val=None, dtype=np.float32, name="Filter2nd"):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self.A, self.B = self._construct_matrices(tau, gain)
        self.init_val = init_val

        # states
        self.x = np.zeros(2, dtype=dtype)
        self.dx = np.zeros_like(self.x)

    def __call__(self, action):
        action = np.asarray(action).astype(self.dtype)
        fn = lambda x: x.dot(self.A) + action.dot(self.B)
        self.dx = no_time_rungekutta(fn, self.dt, self.x)
        self.x += self.dx * self.dt
        return self.get_state()

    def reset(self, init_val=None):
        init_val = init_val if init_val is not None else self.init_val
        self.x = self._make_init_vector(init_val)
        self.dx = np.zeros_like(self.x)
        return self.x

    def get_state(self):
        return self.x.astype(self.dtype)

    def get_full_state(self):
        xs = np.concatenate([self.x, self.dx[1:2]], axis=0)
        return xs.astype(self.dtype)

    def _make_init_vector(self, init_val=None):
        if init_val is None:
            return np.zeros_like(self.x).astype(self.dtype)
        if not hasattr(init_val, "__len__"):
            return np.array([init_val, 0])
        init_val = np.asarray(init_val)
        assert init_val.shape == self.x.shape
        return np.asanyarray(init_val).astype(self.dtype)

    def _construct_matrices(self, t, k):
        it = 1 / t
        it2 = it * it
        A = np.array([
            [0, 1],
            [-it2, -2 * it]
        ], dtype=self.dtype).T
        B = np.array([0, k * it2], dtype=self.dtype).T
        return A, B
