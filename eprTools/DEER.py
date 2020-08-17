from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, fminbound, least_squares
import matplotlib.pyplot as plt
from eprTools.utils import homogeneous_nd, read_param_file, generate_kernel, reg_range, reg_operator, cvxnnls
from eprTools.SVP import SVP


class DEERSpec:
    def __init__(self, time, spec, **kwargs):
        # Working values
        self.time = time
        self.spec = spec / max(spec)
        self.real = self.spec.real
        self.imag = self.spec.imag

        # Fit results
        self.fit = None
        self.alpha = None
        self.P = None
        self.s_error = np.inf

        # L-curve misc
        self.params = [0, 0]
        self.alpha_idx = None
        self.rho = None
        self.eta = None

        # Kernel parameters
        self.K = None
        r = kwargs.get('r', (15, 80))
        if len(r) == 2:
            self.r = np.linspace(r[0], r[1], len(self.time))
        else:
            self.r = r

        self.L = reg_operator(self.r)

        # Default phase and trimming parameters
        self.trim_length = None
        self.user_trim = False

        self.phi = None
        self.user_phi = False
        self.do_phase = kwargs.get('do_phase', True)

        self.zt = None
        self.user_zt = False
        self.do_zero_time = kwargs.get('do_zero_time', True)

        self.L_criteria = kwargs.get('L_criteria', 'aic')

        # Default background correction parameters
        self.bg_model = homogeneous_nd
        self.background = None
        self.background_param = None

        # Raw Data Untouched
        self.raw_time = time
        self.raw_spec = spec
        self.raw_real = spec.real
        self.raw_imag = spec.imag

        self._update()

    @classmethod
    def from_file(cls, file_name, r=(15, 80)):

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
            spec = spec[10:]
            # Delete zeros at end of array (uncompleted scans)
            spec = spec[~np.all(spec == 0, axis=1)]

            # Roll with the mean for now.
            # TODO: Implement usage of individual scans in the future.
            spec = spec.mean(axis=0)

        # Construct DEERSpec object
        return cls(time, spec, r=r, do_phase=do_phase)

    @classmethod
    def from_array(cls, time, spec, r=(15, 80), do_phase=False, do_zero_time=False):
        spec = np.asarray(spec, dtype=complex)

        # Combine real and imaginary
        # Construct DEERSpec object

        return cls(time, spec, r=r,
                   do_phase=do_phase, do_zero_time=do_zero_time)

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

        spec_real = K.dot(P)
        spec_imag = np.zeros_like(spec_real)

        return cls(time, spec_real, spec_imag, background_kind='3D', background_k=0,
                   r=r, do_phase=False, do_zero_time=False)

    def __eq__(self, spc):
        if not isinstance(spc, DEERSpec):
            return False
        elif np.any(spc.raw_time != self.raw_time):
            return False
        elif np.any(spc.raw_spec_real != self.raw_spec_real):
            return False
        elif np.any(spc.raw_spec_imag != self.raw_spec_imag):
            return False
        elif np.any(spc.K != self.K):
            return False
        elif self.alpha != self.alpha:
            return False
        else:
            return True

    def copy(self):
        """
        Create a deep copy of the DEERSpec object.
        """
        return deepcopy(self)

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

            def get_imag_norm_squared(phi):
                spec_imag = np.imag(fit_set * np.exp(1j * phi))
                return spec_imag @ spec_imag

            # Find Phi that minimizes norm of imaginary data
            phi = minimize(get_imag_norm_squared, phi0)
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
        real = interp1d(self.time, self.real)(time)

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
        self.spec /= self.spec[np.argmin(np.abs(self.time))]
        self.real = self.spec.real
        self.imag = self.spec.imag

    def residual(self, params, fit_alpha=False):
        # Get dipolar kernel
        K, r, t = generate_kernel(self.r, self.time)

        # Add background to kernel
        self.lam = params[0]
        self.background = self.bg_model(self.time, *params[1:])

        self.K = (1 - self.lam + self.lam * K) * self.background[:, None]

        diff = np.abs(self.params - params) / params
        if np.all(diff < 1e-1) and not fit_alpha:
            self.get_score(self.alpha)
        else:
            self.alpha_range = np.log10(reg_range(self.K, self.L))
            self.alpha = fminbound(self.get_score, min(self.alpha_range), max(self.alpha_range), xtol=0.01)

        self.params = params.copy()
        return self.s_error

    def get_fit(self):

        #Intelligent fist guesses
        lam0 = (self.real.max() - self.real.min()) * 2 / 3
        least_squares(self.residual, x0=(lam0, 1e-4), bounds=([0., 0.], [1., 1e-3]), ftol=1e-9)
        #SVP(self.residual, x0=(lam0, 1e-4), lb=(0., 0.), ub=(1., 1e-2), ftol=1e-9, xtol=1e-9)

        self.residual(self.params, fit_alpha=True)

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

        self.P = cvxnnls(self.real, self.K, self.L, alpha)
        self.fit = self.K @ self.P
        self.s_error = self.real - self.fit

        if self.L_criteria == 'gcv':
            return self.get_GCV_score(alpha, self.s_error)
        elif self.L_criteria == 'aic':
            return self.get_AIC_score(alpha, self.s_error)

    def get_AIC_score(self, alpha, s_error):
        """
        Gets the Akaike Information Critera (AIC) score for a given fit with a given smoothing parameter.

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the AIC score for the given alpha and fit
        """
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)
        if np.trace(H_alpha) < 0:
            return np.inf
        nt = self.K.shape[1]
        score = nt * np.log((s_error @ s_error) / nt) + (2 * np.trace(H_alpha))

        return score

    def get_GCV_score(self, alpha, s_error):
        """
        Gets the Generalized cross validation (GCV) score for a given fit with a given smoothing parameter.

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the GCV score for the given alpha and fit
        """
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)
        if np.trace(H_alpha) < 0:
            return np.inf
        nt = self.K.shape[1]
        score = (s_error @ s_error) / ((1 - np.trace(H_alpha) / nt) ** 2)
        return score
