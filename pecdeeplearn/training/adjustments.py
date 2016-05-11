from __future__ import division

import numpy as np


class ParameterAdjuster:
    def __init__(self, name, start, stop):
        self.name = name
        self.start = start
        self.stop = stop
        self.ls = None

    def __call__(self, net, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, net.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(net, self.name).set_value(new_value)