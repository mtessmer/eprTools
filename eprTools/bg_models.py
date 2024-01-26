import numpy as np


class HomogeneousND:

    def __init__(self, t, d=3):
        self.t = t
        self.dimensions = d
        self.default_params = [1e-3]

    def __call__(self, params):
        return np.exp(-params[0] * (np.abs(self.t) ** (self.dimensions / 3)))


class Polynomial:

    def __init__(self, t, order=3):
        self.t = t
        self.order = order
        self.default_params = [1, 1, 1, 1]

    def __call__(self, params):
        return np.polyval(params, self.t)

class Flat:

    def __init__(self, t):
        self.t = t
        self.default_params = []

    def __call__(self, params):
        return np.ones(len(self.t))