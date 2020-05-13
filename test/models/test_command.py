# coding: utf-8

import numpy as pn
import pandas as pd
from matplotlib import pyplot as plt
from xsim.time import *
from xsim.models.command import *
from xsim.models.filter import *
from xsim.logger import *


def test_PoissonRectangularCommand():
    print("run test for PoissonRandomRectangularCommand")

    dt = 0.1
    due = 10.0

    max_amp = 2.0
    interval = 1.0
    tau = 0.1

    cmd = PoissonRectangularCommand(max_amp, interval)
    fil = Filter2nd(dt, tau)

    log = Logger()

    time = 0
    r = cmd.reset()
    r = fil.reset(init_val=r)
    log.store(time=time, cmd=r).flush()

    for time in generate_step_time(due, dt):
        r = cmd(time)
        r = fil(r)
        log.store(time=time, cmd=r).flush()

    result = Retriever(log)
    result = pd.DataFrame({
        "time": result.time(),
        "command": result.cmd(0)
    })
    result.plot(x="time", y="command")
    plt.show()


if __name__ == '__main__':
    test_PoissonRectangularCommand()
