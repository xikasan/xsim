# coding: utf-8

from xsim.time import generate_step_time
from xsim.models.filter import Filter2nd
from xsim.logger import Logger


def run():
    print("run test for logger")
    dt = 0.1
    due = 10
    tau = 1.0

    filt = Filter2nd(dt, tau)

    r  = 1.0
    xs = filt.reset()
    print("time:{:4.2f} xs:{:6.4f}".format(0, xs[0]))
    for time in generate_step_time(due, dt):
        xs = filt(r)
        print("time:{:4.2f} xs:{:6.4f}".format(time, xs[0]))


if __name__ == '__main__':
    run()
