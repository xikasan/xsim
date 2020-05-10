# coding: utf-8

import xsim


def run():
    print("run test for logger")
    dt = 0.1
    due = 10
    tau = 1.0

    filt = xsim.Filter2nd(dt, tau)

    logger = xsim.Logger()

    r  = 1.0
    xs = filt.reset()
    time = 0.0
    logger.store(time=time, xs=xs).flush()
    print("time:{:4.2f} xs:{:6.4f}".format(time, xs[0]))
    for time in xsim.generate_step_time(due, dt):
        xs = filt(r)
        logger.store(time=time, xs=xs).flush()
        print("time:{:4.2f} xs:{:6.4f}".format(time, xs[0]))

    result = xsim.Retriever(logger)
    print(result.time())


if __name__ == '__main__':
    run()
