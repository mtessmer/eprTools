import numpy as np
from .utils import generate_kernel


class DeerModel:

    def __init__(self, r, t, bg_model, K_kwargs=None):
        self.K_kwargs = K_kwargs or {}
        self._K, self._r, self._t = generate_kernel(r, t, **self.K_kwargs)
        self.bg_model = bg_model
        self.default_params = [0.5] + self.bg_model.default_params

    def __call__(self, params):
        return (1 - params[0] + params[0] * self.K) * self.bg_model(params[1:])[:, None]

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t):
        self.bg_model.t = new_t
        self._K, self._r, self._t = generate_kernel(self.r, new_t, **self.K_kwargs)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, new_r):
        self._K, self._r, self._t = generate_kernel(new_r, self.t, **self.K_kwargs)

    @property
    def K(self):
        return self._K