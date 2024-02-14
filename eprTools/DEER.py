from copy import deepcopy
import numbers, pickle, inspect
from typing import Sized
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, fminbound, least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from .nnls_funcs import NNLS_FUNCS
from .selection_methods import SELECTION_METHODS
from .utils import *
from .fg_models import DeerModel
from .bg_models import HomogeneousND
from tqdm import tqdm
eps = np.finfo(float).eps

class DeerExp:
    def __init__(self, time, V, **kwargs):

        # Raw Data Untouched
        self.raw_time = time.copy()
        self.raw_V = V.copy()
        self.raw_real = V.real.copy()
        self.raw_imag = V.imag.copy()

        # Working values
        self.time = time
        self.V = V

        # Default phase and trimming parameters
        self.trim_length = None
        self.user_trim = False

        self.phi = None
        self.user_phi = False
        self.do_phase = kwargs.get('do_phase', False)

        self.t0 = None
        self.user_zt = False
        self.do_zero_time = kwargs.get('do_zero_time', True)
        self.do_A0 = kwargs.get('do_A0', True)

        # Fit results
        self.nnls = kwargs.get('nnls', 'cvxnnls')
        self.Vfit = None
        self.alpha = None
        self._P = kwargs.get('P', None)
        self.residuals = np.inf

        self.K_kwargs = {k: v for k, v in kwargs.items() if k in ['g']}

        self.mod_penalty = kwargs.get('mod_penalty', 0.25)
        self.freeze_alpha = False
        self.freeze_mod = False

        # Model
        self._r = setup_r(kwargs.get('r', (15, 80)), kwargs.get('size', len(self.raw_time)))

        self.bg_model = kwargs.get('bg_model', HomogeneousND(self.time))
        if inspect.isclass(self.bg_model):
            self.bg_model = self.bg_model(self.time)


        self.model = kwargs.get('fg_model', DeerModel(self.r, self.time, self.bg_model, self.K_kwargs))
        if inspect.isclass(self.bg_model):
            self.bg_model = self.bg_model(self.r, self.time, self.bg_model)

        self.params = self.model.default_params
        self.background = None

        # Regularization
        self.L = reg_operator(self.r, kind='L2')
        self.selection_method = kwargs.get('L_criteria', 'gcv')

        self.alpha_idx = None
        self.rho = None
        self.eta = None

        self._update()

    @classmethod
    def from_file(cls, file_name, r=(15, 80), **kwargs):

        t, V, params = read_bruker(file_name, return_params=True)

        if kwargs.pop('do_phase', True):
            V = opt_phase(V)
            do_phase = False

        if len(V.shape) > 1:
            V = V.sum(axis=0)

        # Construct DeerExp object
        return cls(t, V, r=r, do_phase=do_phase, **kwargs)

    @classmethod
    def from_array(cls, time, V, r=(15, 80), **kwargs):
        """
        Create a DEERExp object from an array like data structure.

        :param time: np.ndarray
            Time axis of the DEER data
        :param V: np.ndarray
            DEER Data array, can be real or complex valued
        :param r: float, 2-tuple, np.ndarray-like
            desired distance access, float --> Upper bound, 2-tuple --> (Lower bound, Upper bound), array-like --> axis

        :return: DEERExp
            A DEERExp object with the user supplied data
        """

        V = np.asarray(V, dtype=complex)
        return cls(time, V, r=r, **kwargs)

    @classmethod
    def from_distribution(cls, r, P, time=3500):
        """
        Constructor method for

        :param r: ndarray
            Distance coordinates of the distance distribution
        :param P: ndarray
            Probability density of the distribution corresponding to the distance coordinate array
        :return:
        """
        if len(r) != len(P):
            raise ValueError('r and P must have the same number of points')

        K, r, time = generate_kernel(r, time, size=len(P))

        V = K.dot(P)
        V = np.asarray(V, dtype=complex)

        return cls(time, V, background_kind='3D', background_k=0,
                   r=r, do_zero_time=False, P=P)

    @property
    def real(self):
        return self.V.real

    @property
    def imag(self):
        return self.V.imag

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        if isinstance(value, numbers.Number):
            self._r = np.linspace(0, value, len(self.time))
        elif isinstance(value, Sized):
            if len(value) == 2:
                self._r = np.linspace(*value, len(self.time))
            elif len(value) == 3:
                self._r = np.linspace(*value)
            else:
                self._r = np.asarray(value)
        self._update()

    @property
    def nnls(self):
        return self._nnls

    @nnls.setter
    def nnls(self, value):
        if value in NNLS_FUNCS.keys():
            self._nnls=NNLS_FUNCS[value]
        elif callable(value):
            self._nnls = value
        else:
            raise KeyError(f"{value} is not a recognized nnls method or function")

    @property
    def selection_method(self):
        return self._selection_method

    @selection_method.setter
    def selection_method(self, value):
        if isinstance(value, str):
            value = value.lower()

        if value in SELECTION_METHODS.keys():
            self._selection_method = SELECTION_METHODS[value]
        elif callable(value):
            self._selection_method = value
        else:
            raise KeyError(f"{value} is not a recognized selection method or function")

    def __eq__(self, spc):
        if not isinstance(spc, DeerExp):
            return False
        elif not np.allclose(spc.time, self.time):
            return False
        elif not np.allclose(spc.V, self.V):
            return False
        else:
            return True

    def copy(self):
        """
        Create a deep copy of the DEERExp object.
        """
        return deepcopy(self)

    def save(self, filename='DEERExp.pkl'):
        """
        Save DEERExp object as pickle file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def _update(self):
        """Update all self variables except fit. internal use only."""

        if self.do_zero_time:
            self.zero_time()
        elif self.do_A0:
            _, ft  = fit_zero_time(self.raw_time, self.raw_real, return_fit=True)
            self.A0 = ft[0]
            self.t0 = 0


            # Correct t0 and A0
            self.time = self.raw_time.copy()
            self.V = self.raw_V / self.A0

        if self.do_phase:
            self.phase()

        if self.trim_length is not None:
            self.trim()

        self.model.t = self.time




    def set_trim(self, trim):
        self.trim_length = trim
        self.user_trim = True
        self.t0 = None
        self._update()

    def trim(self):
        """Removes last N points of the deer trace. Used to remove noise explosion and 2+1 artifacts."""
        mask = self.time < self.trim_length
        self.time = self.time[mask]
        self.V = self.V[mask]

    def set_phase(self, phi):
        if np.abs(phi) > 2 * np.pi:
            phi = np.deg2rad(phi)

        self.phi = phi
        self.user_phi = True
        self.do_phase = True
        self._update()

    def phase(self):
        """set phase to maximize signal in real component and minimize signal in imaginary component"""
        if self.phi is None:
            # Find Phi that minimizes norm of imaginary data
            self.V, self.phi = opt_phase(self.V, return_params=True)

        else:
            # Use existing phi
            self.V = self.V * np.exp(1j * self.phi)

    def set_zero_time(self, zero_time):
        self.user_zt = True
        self.t0 = zero_time
        self.do_zero_time = True
        self._update()

    def zero_time(self):
        """
        Sets t=0 where 0th moment of a sliding window is closest to 0
        """
        self.A0, self.t0 = fit_zero_time(self.raw_time, self.raw_real, return_params=True)

        # Correct t0 and A0
        self.time = self.raw_time - self.t0
        if self.do_A0:
            self.V = self.raw_V / self.A0

    def get_model_params(self):


        buffer = int(0.1 * len(self.time))
        # buffer = 2 * np.argmin(savgol_filter(np.diff(self.real), 20, 3))
        bgfit = []
        if isinstance(self.bg_model, HomogeneousND):
            for i in range(buffer * 4):
                # Needs to be able to pass any kw constructor args
                partial_Bfnc = self.bg_model.__class__(self.time[buffer*2 + i:], **self.bg_model.kwargs)
                resid = lambda params : self.real[buffer*2 + i:] - (1-params[0]) * partial_Bfnc(params[1:])
                fit = least_squares(resid, x0=self.model.default_params, bounds=(self.model.lbs, self.model.ubs))
                bgfit.append(fit.x)
            bgfit = np.array(bgfit)
            means = np.mean(bgfit, axis=0)
            stds = np.std(bgfit, axis=0)

        else:
            partial_Bfnc = self.bg_model.__class__(self.time[buffer*2:], **self.bg_model.kwargs)
            resid = lambda params : self.real[buffer*2:] - (1-params[0]) * partial_Bfnc(params[1:])
            fit = least_squares(resid, x0=self.model.default_params, bounds=(self.model.lbs, self.model.ubs))
            means = fit.x
            stds = np.diag(np.linalg.inv(np.dot(fit.jac.T, fit.jac)))

        self.model.lbs = np.maximum(means - stds, self.model.lbs)
        self.model.ubs = np.minimum(means + stds, self.model.ubs)
        self.model.par0 = np.maximum(np.minimum(self.model.ubs, means), self.model.lbs)

    def get_fit(self):

        self.get_model_params()
        self.score = np.inf

        opt = least_squares(self.residual, x0=(self.model.par0), bounds=(self.model.lbs, self.model.ubs), ftol=1e-10)
        # self.get_uncertainty()

    def bootstrap(self, n=100):

        noiselvl = np.std(self.real - self.Vfit)
        Ps = []
        def res(param, Vnoise, return_fits = False):
            K = self.model(param)
            P = self.nnls(K, self.L, Vnoise, self.alpha)
            fit = K @ P
            residuals = fit - Vnoise
            if return_fits:
                return P, fit
            else:
                return np.concatenate([residuals, param[0:1]])

        param_list, Ps, fits = [], [], []
        for i in tqdm(range(n)):
            Vnoise = self.real + np.random.normal(0, noiselvl, len(self.real))
            res_v = lambda param : res(param, Vnoise)
            opt = least_squares(res_v, x0=((self.model.par0)), bounds=(self.model.lbs, self.model.ubs))
            # opt = minimize(res_v, x0=(self.mod0, self.bgp0), bounds=((self.lbs[0], self.ubs[0]), (self.lbs[1], self.ubs[1])))
            param_list.append(opt.x)
            P, fit = res(opt.x, Vnoise, return_fits = True)
            Ps.append(P)
            fits.append(fit)



        Ps, fits, param_list = np.array(Ps), np.array(fits), np.array(param_list)
        self.Pstd = np.std(Ps, axis=0)

        Bs = (1 - param_list[:, 0])[:, None] * np.array([self.bg_model(p) for p in param_list[:, 1:]])
        self.Bstd = np.std(Bs, axis=0)

        self.fitstd = np.std(fits, axis=0)



    def residual(self, params, fit_alpha=False):
        # Get dipolar kernel
        K, r, t = generate_kernel(self.r, self.time, **self.K_kwargs)
        # Add background to kernel
        self.lam = params[0]
        self.background = self.bg_model(params[1:])

        self.K = self.model(params)

        diff = np.abs((self.params - params) / params)
        if np.any(diff > 1e-3) or fit_alpha:
            self.alpha_range = (1e-8, 1e4)

            log_alpha = fminbound(lambda x: self.get_score(10**x),
                                  np.log10(min(self.alpha_range)),
                                  np.log10(max(self.alpha_range)), xtol=0.01)

            self.alpha = 10 ** log_alpha

        self.get_score(self.alpha)
        self.params = params.copy()
        return self.regres

    def get_score(self, alpha):
        """
        Helper method to choose score function for evaluating under/overfitting

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the L_curve critera score for the given alpha and fit
        """
        # Get initial matrices of optimization
        self._P = self.nnls(self.K, self.L, self.real, alpha)

        self.Vfit = self.K @ self._P
        self.residuals = self.Vfit - self.real

        mod_penalty = [self.mod_penalty * self.lam] if self.mod_penalty is not None else []
        self.regres = np.concatenate([self.residuals,
                                      alpha * self.L @ self._P, mod_penalty])

        # Regres is not as large as deerlab because self._P is smaller because its per Angstrom not per nm
        self.score = self.selection_method(self.K, self.L, alpha, self.residuals)
        return self.score

    def get_L_curve(self):
        """
        calculate L-curve and return rho, eta and the chosen index

        :return rho, eta, alpha_idx:
        """
        Ps = np.array([self.nnls(self.K, self.L, self.real, alpha) for alpha in self.alpha_range])
        fits = self.K @ Ps.T
        errs = fits - self.real.T[:, None]
        rho = np.log(np.linalg.norm(errs.T, axis=1))
        eta = np.log(np.linalg.norm((self.L @ Ps.T).T, axis=1))
        alpha_idx = np.argmin(np.abs(self.alpha_range - self.alpha))

        return rho, eta, alpha_idx

    def conc(self):
        NA = 6.02214076e23  # Avogadro constant, mol^-1
        muB = 9.2740100783e-24  # Bohr magneton, J/T (CODATA 2018 value)
        mu0 = 1.25663706212e-6  # magnetic constant, N A^-2 = T^2 m^3 J^-1 (CODATA 2018)
        h = 6.62607015e-34  # Planck constant, J/Hz (CODATA 2018)
        ge = 2.00231930436256  # free-electron g factor (CODATA 2018 value)
        hbar = h / 2 / np.pi  # reduced Planck constant, J/(rad/s)
        D = (mu0 / 4 / np.pi) * (muB * ge) ** 2 / hbar  # dipolar constant, m^3 s^-1
        k = self.params[1] * 1000
        conc  = k / (8 * np.pi ** 2 / 9 / np.sqrt(3) * self.lam * D * 1e-6)
        self.conc = conc / (1e-6*1e3*NA)
        return self.conc

    @property
    def P(self):
        return self._P * self.Pscale

    @property
    def Pscale(self):
        return 1 / np.trapz(self._P, self.r)

    def Pci(self, percent):
        alpha = 1 - percent / 100
        p = 1 - alpha/2

        lower_bounds = np.maximum(0, self.P - norm.ppf(p) * self.Pscale * self.Pstd)
        upper_bounds = np.maximum(0, self.P + norm.ppf(p) *  self.Pscale * self.Pstd)
        return lower_bounds, upper_bounds

    @property
    def B(self):
        return (1-self.lam) * self.background

    def Bci(self, percent):
        alpha = 1 - percent / 100
        p = 1 - alpha/2
        lower_bounds = np.maximum(0, self.B - norm.ppf(p) * self.Bstd)
        upper_bounds = np.maximum(0, self.B + norm.ppf(p) * self.Bstd)
        return lower_bounds, upper_bounds

    def fitci(self, percent):
        alpha = 1 - percent / 100
        p = 1 - alpha/2
        lower_bounds = np.maximum(0, self.Vfit - norm.ppf(p) * self.fitstd)
        upper_bounds = np.maximum(0, self.Vfit + norm.ppf(p) * self.fitstd)
        return lower_bounds, upper_bounds