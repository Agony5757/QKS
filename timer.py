# Timer Class
from time import time
import numpy as np

class timer:

    timers   = list()
    messages = list()
    
    def __init__(self):
        pass

    @staticmethod
    def init():
        timer.timer = list()
        timer.messages = list()
        timer.timelog("Start")

    @staticmethod
    def timelog(msg=""):
        timer.timers.append(time())
        timer.messages.append(msg)

    @staticmethod
    def print_elapse(msg="", last_step = False):
        timer.timelog(msg)
 
        print_str = "{} | elpased {}".format(
            timer.messages[-1],
            timer.timers[-1]-timer.timers[-2])

        if last_step is True:
            print_str = "{} -> ".format(timer.messages[-2])+print_str

        print(print_str)      
        return timer.timers[-1]-timer.timers[-2]

    @staticmethod
    def end():
        timer.timelog()
        print("Finished! Total elapsed: {}".format(timer.timers[-1]-timer.timers[0]))
        timer.init()
        return timer.timers[-1]-timer.timers[0]

import numpy as np
if __name__ == "__main__":
    timer.init()
    s = np.zeros((1000000))
    for i in range(1000):
        s[i] += 1
    timer.print_elapse("1e3")

    for i in range(10000):
        s[i] += 1
    timer.print_elapse("1e4")

    for i in range(100000):
        s[i] += 1

    timer.print_elapse("1e5")

    for i in range(1000000):
        s[i] += 1
    timer.print_elapse("1e6")
    timer.end()