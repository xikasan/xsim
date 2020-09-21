# coding: utf-8

from xsim.time import generate_step_time


def run():
    due = 40
    dt = 0.02

    for time in generate_step_time(due, dt):
        print(time)


if __name__ == '__main__':
    run()
