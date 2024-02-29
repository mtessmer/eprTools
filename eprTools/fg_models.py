import numpy as np
from .utils import generate_kernel


class DeerModel:

    def __init__(self, r, t, bg_model, K_kwargs=None):
        self.K_kwargs = K_kwargs or {}
        self._K, self._r, self._t = generate_kernel(r, t, **self.K_kwargs)
        self.bg_model = bg_model
        self._default_params = [0.5]
        self._lbs = [1e-3]
        self._ubs = [1]

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

    @property
    def lbs(self):
        return np.concatenate([self._lbs, self.bg_model.lbs])

    @lbs.setter
    def lbs(self, new_lbs):
        self._lbs = new_lbs[:len(self._lbs)]
        self.bg_model.lbs = new_lbs[len(self._lbs):]

    @property
    def ubs(self):
        return np.concatenate([self._ubs, self.bg_model.ubs])

    @ubs.setter
    def ubs(self, new_ubs):
        self._ubs = new_ubs[:len(self._ubs)]
        self.bg_model.ubs = new_ubs[len(self._ubs):]

    @property
    def default_params(self):
        default_params = self._default_params + self.bg_model.default_params
        return default_params

    @default_params.setter
    def default_params(self, new_values):
        self._default_params = new_values[:len(self._default_params)]
        self.bg_model.default_params = new_values[len(self._default_params):]

    @property
    def initial_params(self):
        if not hasattr(self, "_initial_params"):
            self._initial_params = self.default_params

        return self._initial_params

    @initial_params.setter
    def initial_params(self, new_values):
        assert len(new_values) == len(self._initial_params)
        self._initial_params = new_values
