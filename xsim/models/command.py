# coding: utf-8

import numpy as np


def pulse(time, period, amplitude, bias=None):
    ret = amplitude if (time % period) < (period / 2) else -amplitude
    if bias is not None:
        ret += bias
    return ret


class BaseCommand:

    def __init__(self, amplitude=1.0, bias=0.0, dtype=np.float32, name="Command"):
        self.amplitude = np.asarray(amplitude).astype(dtype)
        self.bias = np.asarray(bias).astype(dtype)
        self.dtype = dtype

    def __call__(self, time):
        raise NotImplementedError()


class RectangularCommand(BaseCommand):

    def __init__(self, period=1.0, name="RectangularCommand", **kwargs):
        super().__init__(name=name, **kwargs)

        self.period = period

    def __call__(self, time):
        cmd = pulse(time, self.period, self.amplitude, bias=self.bias)
        return self.dtype(cmd)


class PoissonRectangularCommand(BaseCommand):

    def __init__(self, max_amplitude, interval, name="PoissonRectangularCommand", **kwargs):
        super().__init__(name=name, **kwargs)

        self.max_amplitude = max_amplitude
        self.rate = 1.0 / interval

        self.next_time = 0.0
        self.amplitude = 0.0
        self.generate_next()

    def __call__(self, time):
        if time >= self.next_time:
            self.generate_next()
        return self.amplitude

    def reset(self):
        self.next_time = 0.0
        self.amplitude = 0.0
        self.generate_next()
        return self.amplitude

    def generate_next(self):
        self.next_time -= np.log(np.random.rand()) / self.rate
        self.amplitude = self.generate_clipped_normal() * self.max_amplitude

    def generate_clipped_normal(self, max_try=10):
        for _ in range(max_try):
            norm = self.dtype(np.random.normal(0, 0.5))
            if -1 <= norm <= 1:
                return norm
        return np.clip(norm, -1, 1)
