# coding: utf-8

import pandas as pd
from matplotlib import pyplot as plt
from xsim.time import *
from xsim.models.transferfunc import *
from xsim.logger import Logger, Retriever


def test_tf2():
    dt = 0.02
    due = 10.0

    tf = TransferFunc2nd(dt, [3, 2], [1, 0])
    logger = Logger()

    # pre simulation log
    time = 0
    xs = tf.get_state()
    logger.store(time=time, state=xs).flush()

    for time in generate_step_time(due, dt):
        xs = tf(1)
        logger.store(time=time, state=xs).flush()

    result = Retriever(logger)
    result = pd.DataFrame({
        "time": result.time(),
        "response": result.state(idx=0)
    })

    result.plot(x="time", y="response")
    plt.show()


if __name__ == '__main__':
    test_tf2()
