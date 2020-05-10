# coding: utf-8

from xsim.time import generate_step_time
from xsim.models.filter import Filter1st, Filter2nd


def run_filter1st():
    print("run test for Filter1st")
    dt = 0.1
    due = 10
    tau = 1.0
    filt = Filter1st(dt, tau)

    r = 1.0
    x = filt.reset()
    print("time:{:4.2f} x:{:6.4f}".format(0, x))
    for time in generate_step_time(due, dt):
        x = filt(r)
        print("time:{:4.2f} x:{:6.4f}".format(time, x))


def run_filter2nd():
    print("run test for Filter2nd")
    dt = 0.1
    due = 10
    tau = 1.0

    filt = Filter2nd(dt, tau)
    filc = Filter1st(dt, tau)

    r = 1.0
    x  = filc.reset()
    xs = filt.reset()
    print("time:{:4.2f} xs:{:6.4f} x:{:6.4f}".format(0, xs[0], x))
    for time in generate_step_time(due, dt):
        x  = filc(r)
        xs = filt(r)
        print("time:{:4.2f} xs:{:6.4f} x:{:6.4f}".format(time, xs[0], x))


if __name__ == '__main__':
    run_filter2nd()
