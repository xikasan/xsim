# coding: utf-8

import xsim
import xtools as xt


def run():
    print("run test for logger")
    dt = 0.1
    due = 10
    tau = 1.0

    filt = xsim.Filter2nd(dt, tau)

    logger = xsim.Logger()

    r  = 1.0
    xs = filt.reset()
    for time in xsim.generate_step_time(due, dt):
        xs = filt(r)
        logger.store(time=time, xs=xs).flush()
        print("time:{:4.2f} xs:{:6.4f}".format(time, xs[0]))

    result = xsim.Retriever(logger)
    print(result.time())
    print(result.time(fn=xt.d2r))


if __name__ == '__main__':
    run()
