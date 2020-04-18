# coding: utf-8

from xsim.time import generate_step_time
from xsim.models.filter import Filter1st


def run():
    print("run")
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



if __name__ == '__main__':
    run()
