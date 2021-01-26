from copy import deepcopy
import numbers, pickle
from collections import Sized
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, fminbound, least_squares
from scipy.optimize._numdiff import approx_derivative
import matplotlib.pyplot as plt
from eprTools.utils import homogeneous_nd, read_param_file, generate_kernel, reg_range
from eprTools.utils import reg_operator, hccm, get_imag_norm_squared, multi_phase
from eprTools.nnls_funcs import NNLS_FUNCS
from eprTools.selection_methods import SELECTION_METHODS
from .SVP import SVP

class DEERSpec:
    def __init__(self, time, spec, **kwargs):
        # Working values
        self.time = time
        self.spec = spec / max(spec)
        self.real = self.spec.real
        self.imag = self.spec.imag

        # Default phase and trimming parameters
        self.trim_length = None
        self.user_trim = False

        self.phi = None
        self.user_phi = False
        self.do_phase = kwargs.get('do_phase', True)

        self.zt = None
        self.user_zt = False
        self.do_zero_time = kwargs.get('do_zero_time', True)

        # Fit results
        self.nnls = kwargs.get('nnls', 'cvxnnls')
        self.fit = None
        self.alpha = None
        self.P = kwargs.get('P', None)
        self.residuals = np.inf

        # L-curve misc
        self.params = [0, 0]
        self.alpha_idx = None
        self.rho = None
        self.eta = None

        # Kernel parameters
        self.K = None
        self.r = kwargs.get('r', (15, 80))
        self.L = reg_operator(self.r, type='L2+')
        self.selection_method = kwargs.get('L_criteria', 'aic')

        # Default background correction parameters
        self.bg_model = homogeneous_nd
        self.background = None

        # Raw Data Untouched
        self.raw_time = time
        self.raw_spec = spec
        self.raw_real = spec.real
        self.raw_imag = spec.imag

        self._update()

    @classmethod
    def from_file(cls, file_name, r=(15, 80), **kwargs):

        # Look for time from DSC file
        param_file = file_name[:-3] + 'DSC'
        try:
            param_dict = read_param_file(param_file)
        except OSError:
            print("Warning: No parameter file found")

        # Don't perform phase correction if experiment was not phase cycled
        do_phase = True
        if param_dict['PlsSPELLISTSlct'][0] == 'none':
            do_phase = False

        # Calculate time axis data from experimental params
        points = int(param_dict['XPTS'][0])
        time_min = float(param_dict['XMIN'][0])
        time_width = float(param_dict['XWID'][0])

        if 'YPTS' in param_dict.keys():
            n_scans = int(param_dict['YPTS'][0])
        else:
            n_scans = 1

        time_max = time_min + time_width
        time = np.linspace(time_min, time_max, points)

        # Read spec data
        spec = np.fromfile(file_name, dtype='>d')

        # Reshape and form complex array
        spec.shape = (-1, 2)
        spec = spec[:, 0] + 1j * spec[:, 1]

        # Drop first ten scans because for some reason our spec phase shifts during early scans
        # TODO: Impliment this as an option because I can't imagine that all specs do this
        if n_scans > 1:
            spec.shape = (n_scans, -1)
            #spec = spec[10:]
            # Delete zeros at end of array (uncompleted scans)
            spec = spec[~np.all(spec == 0, axis=1)]
            spec = multi_phase(spec)

            do_phase = False

            # Roll with the mean for now.
            # TODO: Implement usage of individual scans in the future.
            spec = spec.mean(axis=0)


        # Construct DEERSpec object
        return cls(time, spec, r=r, do_phase=do_phase, **kwargs)

    @classmethod
    def from_array(cls, time, spec, r=(15, 80)):
        """
        Create a DEERSpec object from an array like data structure.

        :param time: np.ndarray
            Time axis of the DEER data
        :param spec: np.ndarray
            DEER Data array, can be real or complex valued
        :param r: float, 2-tuple, np.ndarray-like
            desired distance access, float --> Upper bound, 2-tuple --> (Lower bound, Upper bound), array-like --> axis

        :return: DEERSpec
            A DEERSpec object with the user supplied data

        >>>
        >>>
        >>>
        >>>
        """

        spec = np.asarray(spec, dtype=complex)
        return cls(time, spec, r=r, do_zero_time=False)

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
        K, r, time = generate_kernel(r, time, size=len(P))

        spec = K.dot(P)
        spec = np.asarray(spec, dtype=complex)

        return cls(time, spec, background_kind='3D', background_k=0,
                   r=r, do_phase=False, do_zero_time=False, P=P)

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
        if not isinstance(spc, DEERSpec):
            return False
        elif not np.allclose(spc.time, self.time):
            return False
        elif not np.allclose(spc.spec, self.spec):
            return False
        else:
            return True

    def copy(self):
        """
        Create a deep copy of the DEERSpec object.
        """
        return deepcopy(self)

    def save(self, filename='DEERSpec.pkl'):
        """
        Save DEERSpec object as pickle file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def _update(self):
        """
        Update all self variables except fit. internal use only.
        """
        if self.trim_length is not None:
            self.trim()

        if self.do_phase:
            self.phase()

        if self.do_zero_time:
            self.zero_time()

    def set_trim(self, trim):
        self.trim_length = trim
        self.user_trim = True
        self.zt = None
        self._update()

    def trim(self):
        """
        Removes last N points of the deer trace. Used to remove noise explosion and 2+1 artifacts.
        """
        mask = self.raw_time < self.trim_length
        self.time = self.raw_time[mask]
        self.spec = self.raw_spec[mask]

    def set_phase(self, phi):
        if np.abs(phi) > 2 * np.pi:
            phi = np.deg2rad(phi)

        self.phi = phi
        self.user_phi = True
        self._update()

    def phase(self):
        """
        set phase to maximize signal in real component and minimize signal in imaginary component
        """
        if self.phi is None:
            # Initial guess for phase shift
            phi0 = np.arctan2(self.imag[-1], self.real[-1])

            # Use last 7/8ths of data to fit phase
            fit_set = self.spec[int(round(len(self.spec) / 8)):]

            # Find Phi that minimizes norm of imaginary data
            phi = minimize(get_imag_norm_squared, (fit_set, phi0))
            phi = phi.x
            spec = self.spec * np.exp(1j * phi)

            # Test for 180 degree inversion of real data
            if np.real(spec).sum() < 0:
                phi = phi + np.pi

            self.phi = phi

        # Adjust phase
        self.spec = self.spec * np.exp(1j * self.phi)

    def set_zero_time(self, zero_time):
        self.user_zt = True
        self.zt = zero_time
        self._update()

    def zero_time(self):
        """
        Sets t=0 where 0th moment of a sliding window is closest to 0
        """
        time = np.arange(self.time.min(), self.time.max() + 1)
        real = interp1d(self.time, self.spec.real)(time)

        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)

            if len(xData) != len(data):
                xData = xData[:-1]

            return np.dot(data, xData)

        if not self.zt:
            # make zero_moment of all windows tx(spec_max_idx)/2 and find minimum
            spec_max_idx = real.argmax()
            half_spec_max_idx = int(spec_max_idx / 2)
            low_moment = np.inf

            # search near spec_max_idx for min(abs(int(frame)))
            for i in range(spec_max_idx - half_spec_max_idx, spec_max_idx + half_spec_max_idx):
                lFrame = i - half_spec_max_idx
                uFrame = i + half_spec_max_idx + 1
                test_moment = zero_moment(self.real[lFrame: uFrame])

                if abs(test_moment) < abs(low_moment):
                    low_moment = test_moment
                    spec_max_idx = i

            self.zt = time[spec_max_idx]

        self.time -= self.zt

        # Normalize values
        zero_val = self.spec[np.argmin(np.abs(self.time))]
        self.spec /= np.maximum(abs(zero_val.real), abs(zero_val.imag))
        self.real = self.spec.real
        self.imag = self.spec.imag

    def get_kernel(self, params):
        K, r, t = generate_kernel(self.r, self.time)

        # Add background to kernel
        lam = params[0]
        background = self.bg_model(self.time, *params[1:])

        K = (1 - lam + lam * K) * background[:, None]
        return K

    def residual(self, params, fit_alpha=False):
        # Get dipolar kernel
        K, r, t = generate_kernel(self.r, self.time)

        # Add background to kernel
        self.lam = params[0]
        self.background = self.bg_model(self.time, *params[1:])

        self.K = (1 - self.lam + self.lam * K) * self.background[:, None]

        diff = np.abs(self.params - params) / params
        if np.any(diff > 1e-3) or fit_alpha:
            self.alpha_range = reg_range(self.K, self.L)
            self.alpha = fminbound(self.get_score, min(self.alpha_range), max(self.alpha_range), xtol=0.01)

        else:
            self.get_score(self.alpha)

        self.params = params.copy()
        return self.regres

    def get_fit(self):
        #Intelligent fist guesses
        lam0 = (self.real.max() - self.real.min()) * 2 / 3
        least_squares(self.residual, x0=(lam0, 1e-4), bounds=([0., 0.], [1., 1e-2]), xtol=1e-10, ftol=1e-10)
        #SVP(self.residual, x0=(lam0, 1e-4), lb=(0., 0.), ub=(1., 1e-2), ftol=1e-9, xtol=1e-9)
        self.get_uncertainty()

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
        self.P = self.nnls(self.K, self.L, self.real, alpha)
        self.fit = self.K @ self.P
        self.residuals = self.fit - self.real
        self.regres = np.concatenate([self.residuals, alpha * self.L @ self.P, alpha * self.L @ self.P])
        return self.selection_method(self.K, self.L, alpha, self.residuals)

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

    def get_uncertainty(self):
        """
        :return:
        """
        # Get jacobian of linear and nonlinear fits
        func = lambda params: self.get_kernel(params) @ self.P
        Jac = np.reshape(approx_derivative(func, self.params), (-1, self.params.size))
        Jac = np.concatenate([Jac, self.K], 1)
        Jreg = self.alpha * self.L
        Jreg = np.concatenate((np.zeros((self.L.shape[0], len(self.params))), Jreg), 1)
        Jac = np.concatenate([Jac, Jreg])


        resreg = self.alpha * self.L @ self.P
        res = np.concatenate([self.residuals, resreg])
        self.covmatrix = hccm(Jac, res, 'HC1')
        std = np.sqrt(np.diag(self.covmatrix))
        self.std = std[2:]
        self.ustd = self.P + self.std
        self.lstd = np.maximum(0, self.P - self.std)

        self.params_std = std[:2]


