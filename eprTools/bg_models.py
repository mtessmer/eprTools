import numpy as np


class HomogeneousND:

    def __init__(self, t, d=3):
        self.t = t
        self.dimensions = d
        self.default_params = [1e-3]

    def __call__(self, params):
        return np.exp(-params[0] * (np.abs(self.t) ** (self.dimensions / 3)))