# coding: utf-8

import numpy as np
from ..rungekutta import no_time_rungekutta
from .base import BaseModel


class TransferFunc1st(BaseModel):

    def __init__(self, dt, a, b, init_val=None, dtype=np.float32, name="TransferFunction1st"):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self.a = a
        self.b = b
        self.init_val = init_val if init_val is not None else 0.0

        # states
        self.x = self.init_val
        self.dx = 0.0

        # state action space
        self.act_low, self.act_high = self.generate_inf_range(1)
        self.obs_low, self.obs_high = self.generate_inf_range(1)

    def __call__(self, u):
        fn = lambda x: self.b * u - self.a * x
        self.dx = no_time_rungekutta(fn, self.dt, self.x)
        self.x += self.dx * self.dt
        return self.get_state()

    def reset(self, init_val=None):
        init_val = init_val if init_val is not None else self.init_val
        self.x = init_val
        self.dx = 0.0
        return self.get_state()

    def get_state(self):
        return self.dtype(self.x)

    def get_full_state(self):
        xs = np.array([self.x, self.dx], dtype=self.dtype)
        return xs


class TransferFunc2nd(BaseModel):

    def __init__(self, dt, as_, bs_, init_val=None, dtype=np.float32, name="TransferFunction2nd"):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self.A, self.B, self.C = self._construct_matrices(as_, bs_)
        self.init_val = init_val

        # states
        self.x = np.zeros(2, dtype=self.dtype)
        self.dx = np.zeros_like(self.x)

        # state action space
        self.act_low, self.act_high = self.generate_inf_range(2)
        self.obs_low, self.obs_high = self.generate_inf_range(2)

    def __call__(self, u):
        u = np.asarray(u).astype(self.dtype)
        fn = lambda x: x.dot(self.A) + u.dot(self.B)
        self.dx = no_time_rungekutta(fn, self.dt, self.x)
        self.x += self.dt * self.dx
        return self.get_state()

    def reset(self, init_val=None):
        init_val = init_val if init_val is not None else self.init_val
        self.x = init_val
        self.dx = np.zeros_like(init_val)

    def get_state(self):
        return self.x.astype(self.dtype)

    def get_full_state(self):
        xs = np.concatenate([self.x, self.dx[1:]]).astype(self.dtype)
        return xs

    def _construct_matrices(self, as_, bs_):
        A = np.array([
            [0, 1],
            [-as_[1], -as_[0]]
        ], dtype=self.dtype).T
        B = np.array([0, 1], dtype=self.dtype)
        C = np.array([bs_[0], bs_[1]], dtype=self.dtype).T
        return A, B, C
