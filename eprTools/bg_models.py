import numpy as np


class HomogeneousND:

    def __init__(self, t, d=3):
        self.t = t
        self.dimensions = d
        self.default_params = [1e-3]
        self.lbs = [1e-6]
        self.ubs = [1e2]
        self.kwargs = {'d': d}

    def __call__(self, params):
        return np.exp(-params[0] * (np.abs(self.t) ** (self.dimensions / 3)))


class HomogeneousXD:

    def __init__(self, t):
        self.t = t
        self.default_params = [1e-3, 3]
        self.lbs = [1e-6, 0]
        self.ubs = [1e2, 6]
        self.kwargs = {}
    def __call__(self, params):
        return np.exp(-params[0] * (np.abs(self.t) ** (params[1] / 3)))


class Polynomial:

    def __init__(self, t, order=3):
        self.t = t
        self.order = order
        self.default_params = [1, 1, 1, 1]
        self.lbs = [-200, -200, -200, 0]
        self.ubs = [200, 200, 200, 200]
        self.kwargs = {'order': order}

    def __call__(self, params):
        return np.polyval(params, self.t)

class Flat:

    def __init__(self, t):
        self.t = t
        self.default_params = []
        self.kwargs = {}
        self.lbs = []
        self.ubs = []

    def __call__(self, params):
        return np.ones(len(self.t))