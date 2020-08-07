import numbers
import numpy as np
from scipy.optimize import minimize, curve_fit, nnls
from scipy.interpolate import interp1d
from eprTools.utils import homogeneous_3d, read_param_file, fit_nd_background, generate_kernel
import matplotlib.pyplot as plt
import cvxopt as cvo
from time import time
from numba import njit
from copy import deepcopy


class DEERSpec:
    """
    Double elctron-electron resonance spectrum object created from numpy arrays or bruker file.

    ----------
    :param time : ndarray
        Time points of experimental trace in nanoseconds.
    :param spec_real : ndarray
        Real values of experimental DEER trace
    :param spec_imag : ndarray
        Imaginary values of experimental DEER trace
    :param r_min : float, default 15.0
        Minimum distance value in angstroms to include in kernel for fitting P(r). Distance less than r_min will not
        be calculated as part of P(r). r_min should not be below 15 angstroms.
    :param r_max : float, default 80.0
        Maximum distance value in angstroms to include in kernel for fitting P(r). Note that large ranges of the kernel
        increase the condition number making it harder to fit. Additionally large distances can easily be artifacts of
        background correction and should evaluated critically with respect to the length of the time trace.
    :param do_phase : boolean, default True
        Perform phase correction on real and imaginary components fo DEER trace. Not necessary for simulated data.

    Examples
    --------
    Constructing DEERSpec object from a bruker xepr file.

    >>> from eprTools import DEERSpec
    >>> import matplotlib.pyplot as plt
    >>> spc = DEERSpec.from_file('file_name.DTA')
    >>> plt.plot(spc.time spc.real, spc.time, spc.imag)
    >>> plt.show()

    Constructing DEERSpec from ndarray

    >>> from scipy.stats import norm
    >>> from scipy.integrate import trapz
    >>> from eprTools.utils import  generate_kernel
    >>> import numpy as np

    >>> K = generate_kernel(rmin=15, rmax=80, time=5000, size=250)
    >>> rv = norm(loc = 30, scale=3)
    >>> r_min, r_max = 15, 80
    >>> r = np.linspace(r_min, r_max, 250)
    >>> P = rv.pdf(r)
    >>>P = P / trapz(P, r)

    >>> sim_spec = K.dot(P)
    >>> spc = DEERSpec.from_array(time=np.linspace(0, 5000, len(sim_spec)), spec=sim_spec)

    >>> spc.get_fit()
    >>> fig, (ax1, ax2) = plt.subplots(2,1)
    >>> ax1.plot(spc.fit_time, spc.fit, spc.time, spc.real, alpha = 0.5)
    >>> ax2.plot(r, P, spc.r, spc.P / trapz(spc.P, spc.r), alpha = 0.5)
    >>> plt.legend(['actual', 'predicted'])
    >>> plt.show()
    >>> plt.plot()
    """

    def __init__(self, time, spec_real, spec_imag, **kwargs):

        # Working values
        self.time = time
        self.real = spec_real / spec_real.max()
        self.imag = spec_imag / spec_real.max()

        # Tikhonov fit results
        self.fit = None
        self.alpha = None
        self.alpha_idx = None
        self.rho = None
        self.eta = None
        self.P = None

        # Kernel parameters
        self.K = None
        self.r = kwargs.get('r', (15, 80))
        self.fit_time = time

        # Default phase and trimming parameters
        self.phi = None
        self.do_phase = kwargs.get('do_phase', True)
        self.do_zero_time = kwargs.get('do_zero_time', True)
        self.trim_length = None
        self.zt = None
        self.L_criteria = kwargs.get('L_criteria', 'gcv')

        # Default background correction parameters
        self.background_kind = kwargs.get('background_kind', '3D')
        self.background_k = kwargs.get('background_k', 1)
        self.background_fit_t = None

        # Background correction results
        self.form_factor = None
        self.background = None
        self.background_param = None

        # Raw Data Untouched
        self.raw_time = time
        self.raw_spec_real = spec_real
        self.raw_spec_imag = spec_imag

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

        # Separate real and imaginary components of experimental spectrum
        spec_real = spec[0::2]
        spec_imag = spec[1::2]

        # Drop first ten scans because for some reason our spec phase shifts during early scans
        # TODO: Impliment this as an option because I can't imagine that all specs do this
        if n_scans > 1:
            spec_real = spec_real.reshape(n_scans, points)[10:]
            spec_imag = spec_imag.reshape(n_scans, points)[10:]

            # Delete zeros at end of array (uncompleted scans)
            while np.array_equal(spec_real[-1], np.zeros_like(spec_real[-1])):
                spec_real = np.delete(spec_real, -1, 0)
                spec_imag = np.delete(spec_imag, -1, 0)

            spec_real = np.delete(spec_real, -1, 0)
            spec_imag = np.delete(spec_imag, -1, 0)

            # Roll with the mean for now.
            # TODO: Implement usage of individual scans in the future.
            spec_real = spec_real.mean(axis=0)
            spec_imag = spec_imag.mean(axis=0)

        # Construct DEERSpec object
        return cls(time, spec_real, spec_imag, r=r, do_phase=do_phase)

    @classmethod
    def from_array(cls, time, spec_real, spec_imag=None, r=(15, 80), do_zero_time=False):

        do_phase = True
        # Create 0 vector for imaginary component if it is not provided
        if spec_imag is None:
            spec_imag = np.zeros(len(spec_real))

        # Construct DEERSpec object
        return cls(time, spec_real, spec_imag, r=r,
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

        self.trim()

        if self.do_zero_time:
            self.zero_time()

        if self.do_phase:
            self.phase()

        self.correct_background()
        self.compute_kernel()

    def set_kernel_shape(self, shape=250):
        """
        Change the size of the kernel. Does not change the range of time or distance

        :param shape : int, default=250
            length of kernel array. Note that large kernels require more time to fit.

        See Also
        --------
        set_kernel_r

        Examples
        --------

        """

        if isinstance(shape, numbers.Number):
            shape = (shape, shape)

        self.r = np.linspace(min(self.r), max(self.r), shape[1])
        self.fit_time = np.linspace(min(self.time), max(self.time), shape[0])
        self._update()

    def get_L_curve(self, length=80, set_alpha=False):
        """
        Computes and returns L-curve and optimal alpha index

        :param length: int, default 80
            number of alpha values to test when calculating L-curve

        :param set_alpha: boolean, default False
            set object attribute self.alpha after calculating L-curve and determining optimal alpha. Should only be
            required by get_fit()

        :returns rho, eta, alpha_idx: float, float, int
            L-curve points (rho, eta) and optimal smoothing parameter index (alpha_idx)

        See Also
        --------
        get_fit

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from eprTools import DEERSpec
        >>> spc = DEERSpec.from_file('Example_DEER.DTA')
        >>> spc.get_fit()
        >>> rho, eta, idx = spc.get_L_curve()
        >>> plt.plot(rho, eta)
        >>> plt.scatter(rho[idx], eta[idx], c='r')
        >>> plt.show()
        """

        # Test if L-curve has already been calculated
        if self.alpha_idx is not None:
            return self.rho, self.eta, self.alpha_idx

        else:
            # If alpha is already defined construct L-curve centered around it
            if self.alpha:
                log_min = np.logspace(np.log10(self.alpha) - 4, np.log10(self.alpha), int(np.floor(length / 2)))
                log_max = np.logspace(np.log10(self.alpha), np.log10(self.alpha) + 4, int(np.ceil(length / 2)))
                alpha_list = np.concatenate([log_min, log_max])
            # Else center L-curve at 1
            # TODO: implement dynamic choice of L-curve range to account for dynamic kernels
            else:
                alpha_list = np.logspace(-4, 4, length)

            # Preallocate L-Curve data
            rho = np.zeros(len(alpha_list))
            eta = np.zeros(len(alpha_list))
            best_alpha_score = np.inf
            best_alpha = 1

            # Compute (gcv or aic) score of each alpha and store the optimal alpha and score
            for i, alpha in enumerate(alpha_list):
                P, temp_fit = self.get_P(alpha)
                Serr = self.form_factor - temp_fit
                rho[i] = np.log(np.linalg.norm(Serr))
                eta[i] = np.log(np.linalg.norm((np.dot(self.L, P))))
                temp_score = self.get_score(alpha, temp_fit)
                if temp_score < best_alpha_score:
                    best_alpha_score = temp_score
                    best_alpha = alpha

            if set_alpha:
                self.alpha = best_alpha

            # determine closest alpha in to object defined alpha and determine alpha idx
            difference = np.abs(alpha_list - self.alpha)
            self.alpha_idx = np.argmin(difference)
            self.rho = rho
            self.eta = eta

            return self.rho, self.eta, self.alpha_idx

    def set_kernel_r(self, r):

        """
        Set the distance range of the kernel

        :param r_min: int, float, default 15
            minimum distance value of kernel in angstroms

        :param r_max: int, float, default 80
            maximum distance value of kernel in angstroms

        See Also
        --------
        set_kernel_len

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>> import matplotlib.pyplot as plt
        >>> spc = DEERSpec('Example_DEER.DTA')
        >>> spc.get_fit()
        >>> plt.plot(spc.r, spc.P)
        >>> spc.set_kernel_shape(r=(20,60))
        >>> spc.get_fit()
        >>> plt.plot(spc.r, spc.P)
        >>> plt.show()
        """

        self.r = r
        self._update()

    def set_phase(self, phi=0, degrees=True):
        """
        Manually set phase to adjust signal between real and imaginary components

        :param phi: float
            Angle to set the phase to

        :param degrees: bool, default True
            Use unit degrees. If set to False phi will be assumed to be in radians.

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>> import matplotlib.pyplot as plt
        >>> spc = DEERSpec.from_file('Example_DEER.DTA')
        >>> plt.plot(spc.time, spc.real, spc.time, spc.imag, label = ['real', 'imaginary'])
        >>> spc.set_phase(90.0)
        >>> plt.plot(spc.time, spc.real, spc.time, spc.imag, label = ['real', 'imaginary'])
        >>> plt.show()

        """
        if not degrees:
            self.phi = phi
        elif degrees:
            self.phi = np.deg2rad(phi)

        self.do_phase = True

        self._update()

    def set_trim(self, trim=None):
        """
        Sets the time length (in ns) of the experimental form factor. Generally used to remove noisy data at  the
        end of a trace cause by noise explosion or 2+1 artifacts.

        :param trim: int
            length of the desired time trace in nanoseconds

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>> spc = DEERSpec.from_file('examples/Example_DEER.DTA')
        >>> fig, ax = plt.subplots()
        >>> ax.plot(spc.time, spc.real)
        >>> spc.set_trim(2000)
        >>> ax.plot(spc.time, spc.real)
        >>> plt.show()
        """
        self.trim_length = trim
        self.background_fit_t = None
        self._update()

    def set_zero_time(self, zt=None):
        """
        Set the zero time of the form factor.

        :param zt: int,
            new zero time (ns)

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>> spc = DEERSpec.from_file('examples/Example_DEER.DTA')
        >>> fig, ax = plt.subplots()
        >>> ax.plot(spc.time, spc.real)
        >>> spc.set_zero_time(50)
        >>> ax.plot(spc.time, spc.real)
        >>> plt.show()
        """

        self.zt = zt;
        self._update()

    def set_background_correction(self, kind='3D', k=1):
        """
        Perform background correction.

        :param kind: string, default '3D'
            Function used to correct background. Use one of ['3D', '2D', 'poly']

        :param k: int, default 1
            Order used for polynomial fit

        :param fit_time: int, default None
            Length of the experimental trace to be used for fitting the background correction

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>> spc = DEERSpec.from_file('examples/Example_DEER.DTA')
        >>> fig, ax = plt.subplots()
        >>> ax.plot(spc.time, spc.real)
        >>> spc.set_trim(2000)
        >>> ax.plot(spc.time, spc.real)
        >>> plt.show()

        """
        self.background_kind = kind
        self.background_k = k
        self._update()

    def set_L_criteria(self, mode):
        """
        Set the selection criteria for choosing the optimal smoothing parameter (alpha).

        :param mode: string
            Method used to score selection of smoothing parameter. Choose one of ['gcv', 'aic']

                Examples
        --------
        >>> from eprTools import DEERSpec
        >>> spc = DEERSpec.from_file('examples/Example_DEER.DTA')
        >>> print(spc.L_criteria)
        >>> spc.set_L_criteria('aic')
        >>> print(spc.L_criteria)

        """

        self.L_criteria = mode
        self._update()

    def compute_kernel(self):
        """
        Computes kernel in accordance with the default kernel parameters or kernel parameters set with the set helper
        functions. This should only be used internally. See utils.generate_kernel function to construct kernels
        independent of a DEERSpec object.

        See Also
        --------
        set_kernel_len
        set_kernel_r
        utils.generate_kernel
        """
        # Generate new kernel
        self.K, self.r, self.fit_time = generate_kernel(self.r, self.fit_time)

        # Define L matrix
        L = np.zeros((self.K.shape[1] -2, self.K.shape[1]))
        diag = np.arange(self.K.shape[1] - 2)
        L[diag, diag] = 1
        L[diag, diag + 1] = - 2
        L[diag, diag + 2] = 1
        self.L = L

        # Compress Data to kernel dimensions
        f = interp1d(self.time, self.form_factor)
        form_factor = f(self.fit_time)

        if self.background_kind in ['3D', '2D', 'ND']:
            B = homogeneous_3d(np.abs(self.fit_time), self.background_param[0], self.background_param[1], self.d)
        elif self.background_kind == 'poly':
            B = np.polyval(self.background_param, np.abs(self.fit_time))

        self.K = ((1 - self.modulation_depth + self.modulation_depth * self.K).T * B).T

        self.form_factor = form_factor
        self.L = L

    def trim(self):
        """
        Removes last N points of the deer trace. Used to remove noise explosion and 2+1 artifacts.
        """

        self.real = self.raw_spec_real
        self.imag = self.raw_spec_imag
        self.time = self.raw_time

        # normalize working values
        if min(self.real) < 0:
            self.real = self.real - 2 * min(self.real)
            self.imag = self.imag / max(self.real)
            self.real = self.real / max(self.real)

        if not self.trim_length:
            cutoff = None

        else:
            cutoff = self.trim_length
            f_spec_real = interp1d(self.time, self.real, 'cubic')
            f_spec_imag = interp1d(self.time, self.imag, 'cubic')

            self.time = np.arange(self.time.min(), self.time.max())
            self.real = f_spec_real(self.time)
            self.imag = f_spec_imag(self.time)

        self.time = self.time[:cutoff]
        self.real = self.real[:cutoff]
        self.imag = self.imag[:cutoff]

        if self.K is not None:
            size = self.K.shape[1]
        else:
            size = len(self.raw_time)

        self.fit_time = np.linspace(self.time.min(), self.time.max(), size)

    def phase(self):
        """
        set phase to maximize signal in real component and minimize signal in imaginary component
        """

        # Make complex array for phase adjustment
        complex_data = self.real + 1j * self.imag

        if self.phi is None:
            # Initial guess for phase shift
            phi0 = np.arctan2(self.imag[-1], self.real[-1])

            # Use last 7/8ths of data to fit phase
            fit_set = complex_data[int(round(len(complex_data) / 8)):]

            def get_imag_norm_squared(phi):
                spec_imag = np.imag(fit_set * np.exp(1j * phi))
                return np.dot(spec_imag, spec_imag)

            # Find Phi that minimizes norm of imaginary data
            phi = minimize(get_imag_norm_squared, phi0)
            phi = phi.x
            spec_imag = complex_data * np.exp(1j * phi)

            # Test for 180 degree inversion of real data
            if np.real(spec_imag).sum() < 0:
                phi = phi + np.pi

            self.phi = phi

        # Adjust phase
        complex_data = complex_data * np.exp(1j * self.phi)

        # Normalize
        phase_max = np.maximum(np.abs(complex_data.real).max(), np.abs(complex_data.imag).max())
        self.imag = np.imag(complex_data) / phase_max
        self.real = np.real(complex_data) / phase_max

    def zero_time(self):
        """
        Sets t=0 where 0th moment of a sliding window is closest to 0

        """

        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)

            if len(xData) != len(data):
                xData = xData[:-1]

            return np.dot(data, xData)

        # Interpolate data
        f_spec_real = interp1d(self.time, self.real, 'cubic')
        f_spec_imag = interp1d(self.time, self.imag, 'cubic')

        self.time = np.arange(self.time.min(), self.time.max() + 1)
        self.real = f_spec_real(self.time)
        self.imag = f_spec_imag(self.time)

        if not self.zt:
            # ake zero_moment of all windows tx(spec_max_idx)/2 and find minimum
            spec_max_idx = self.real.argmax()
            half_spec_max_idx = int(spec_max_idx / 2)

            lFrame = spec_max_idx - half_spec_max_idx
            uFrame = spec_max_idx + half_spec_max_idx + 1
            low_moment = zero_moment(self.real[lFrame: uFrame])

            # Only look in first 500ns of data
            for i in range(half_spec_max_idx, np.argmin(self.real[:1000]) // 2):
                lFrame = i - half_spec_max_idx
                uFrame = i + half_spec_max_idx + 1

                test_moment = zero_moment(self.real[lFrame: uFrame])

                if abs(test_moment) < abs(low_moment):
                    low_moment = test_moment
                    spec_max_idx = i

        else:
            spec_max_idx = self.zt

        # Adjust time to appropriate zero
        self.fit_time -= self.time[spec_max_idx]
        self.time -= self.time[spec_max_idx]

    def correct_background(self):
        """
        Calculate background fit to be incorporated into the kernel. Previously this method would subtract the
        background from the time domain signal but that functionality has been removed. Future implementations may
        allow for this as well as sqrt(Background) adjustments.
        """

        def score_background(x, d, t):
            k, a = x
            background = homogeneous_3d(t, k, a)
            ffi = self.real - background
            return np.sum(np.abs(ffi))  # @ np.abs(t)

        if self.background_kind in ['3D', '2D']:
            if self.background_kind == '2D':
                self.d = 2
            elif self.background_kind == '3D':
                self.d = 3

            res = minimize(score_background, x0=[1e-5, 0.7], args=(self.d, self.time), bounds=[(1e-7, 0.4), (1e-1, 1)])
            popt = res.x

            self.background = homogeneous_3d(np.abs(self.time), *popt, self.d)
            self.modulation_depth = 1 - popt[1]
            self.background_param = popt
            self.form_factor = self.real

        elif self.background_kind == 'ND':

            res = minimize(score_background, x0=[1e-5, 0.7, 3], args=(self.fit_time),
                           bounds=[(1e-7, 0.4, 0.), (1e-1, 1, 6)])
            popt = res.x

            self.background = homogeneous_3d(np.abs(self.time), *popt)
            self.modulation_depth = 1 - popt[1]
            self.d = popt[2]
            self.background_param = popt
            self.form_factor = self.real

        elif self.background_kind == 'poly':

            popt = np.polyfit(self.time, self.real, deg=self.background_k)
            self.background = np.polyval(popt, self.time)
            self.modulation_depth = 1 - popt[-1]
            self.background_param = popt
            self.form_factor = self.real

    def get_fit(self, alpha=None, true_min=False, fit_method='cvx'):
        """
        Runs fitting protocol with the appropriate settings defined by defaults or user.

        :param alpha: float
            User defined smoothing parameter

        :param true_min: bool
             Decides whether or not to do a true nnls minimization or choose the minimum from a list of possible values

        :param fit_method: string
             Which fitting method to use. Either 'cvx' for convex optimization or 'nnls' for scipy nnls
        """

        self.fit_method = fit_method

        if alpha is None and true_min:
            res = minimize(self.get_score, 1, method='Nelder-Mead')
            self.alpha = res.x

        elif alpha is None:
            self.get_L_curve(set_alpha=True)

        else:
            self.alpha = alpha

        P, self.fit = self.get_P(self.alpha)
        self.P = P / np.sum(P)

    def get_P(self, alpha):
        """
        Calculates the probability distribution for a given Smoothing parameter (Alpha).

        :param alpha: float
            Smoothing parameter
        """

        if self.fit_method == 'nnls':
            C = np.concatenate([self.K, alpha * self.L])
            d = np.concatenate([self.form_factor, np.zeros(shape=self.K.shape[1] - 2)])

            P, _ = nnls(C, d)

        elif self.fit_method == 'cvx':
            P = self.get_P_cvx(alpha)

        else:
            raise NameError('Fit method {} is not supported'.format(self.fit_method))
        fit = self.K.dot(P)
        return P, fit

    def get_P_cvx(self, alpha):
        """
        Convex optimization for NNLS problem. Currently the fastest approach for solving NNLS problem.

        :param alpha: float
            Smoothing parameter
        """
        K = self.K
        L = self.L
        points = self.K.shape[1]

        # Get initial matrices of optimization
        pre_result = (K.T.dot(K) + alpha**2 * L.T.dot(L))

        # get unconstrained solution as starting point.
        P = np.linalg.inv(pre_result).dot(K.T).dot(self.form_factor)
        P = P.clip(min=0)

        B = cvo.matrix(pre_result)

        A = cvo.matrix(-(K.T.dot(self.form_factor.T)))

        # Minimize with CVXOPT constrained to P >= 0
        lower_bound = cvo.matrix(np.zeros(points))
        G = -cvo.matrix(np.eye(points, points))
        cvo.solvers.options['show_progress'] = False
        cvo.solvers.options['abstol'] = 1e-9
        cvo.solvers.options['reltol'] = 1e-8
        P = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))['x']
        P = np.asarray(P).reshape(points, )

        return P

    def get_score(self, alpha, fit=None):

        """
        Helper method to choose score function for evaluating under/overfitting

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the L_curve critera score for the given alpha and fit
        """
        if fit is None:
            P, fit = self.get_P(alpha)

        if self.L_criteria == 'gcv':
            return self.get_GCV_score(alpha, fit)
        elif self.L_criteria == 'aic':
            return self.get_AIC_score(alpha, fit)

    def get_AIC_score(self, alpha, fit):
        """
        Gets the Akaike Information Critera (AIC) score for a given fit with a given smoothing parameter.

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the AIC score for the given alpha and fit
        """
        s_error = self.form_factor - fit
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)
        if np.trace(H_alpha) < 0:
            return np.inf
        nt = self.K.shape[1]
        score = nt * np.log((s_error @ s_error) / nt) + (2 * np.trace(H_alpha))

        return score

    def get_GCV_score(self, alpha, fit):

        """
        Gets the Generalized cross validation (GCV) score for a given fit with a given smoothing parameter.

        :param alpha: float
            Smoothing parameter

        :param fit: numpy ndarray
            The NNLS fit of the the time domain signal for the given alpha parameter.

        :returns score: float
            the GCV score for the given alpha and fit
        """
        s_error = self.form_factor - fit
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)
        if np.trace(H_alpha) < 0:
            return np.inf
        nt = self.K.shape[1]
        score = (s_error @ s_error) / ((1 - np.trace(H_alpha) / nt) ** 2)
        return score


def do_it_for_me(filename, true_min=False, fit_method='cvx'):
    """
    Function to run DEER analysis on experimental data with all defaults

    :param filename: string
        Filename of exerimental data

    :param true_min: bool
             Decides whether or not to do a true nnls minimization or choose the minimum from a list of possible values

    :param fit_method: string
        Which fitting method to use. Either 'cvx' for convex optimization or 'nnls' for scipy nnls
    """

    t1 = time()
    spc = DEERSpec.from_file(filename, r=(15, 120))
    spc.get_fit(true_min=true_min, fit_method=fit_method)
    print('background K', spc.background_param)
    t2 = time()
    print("Fit computed in {}".format(t2 - t1))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 10.5])
    ax1.plot(spc.time, spc.real, label='Experimental')
    ax1.plot(spc.fit_time, spc.fit, label='Fit')
    ax1.plot(spc.time, spc.background, label='background')
    ax2.plot(spc.r, spc.P)

    # Get L-curve
    rho, eta, alpha_idx = spc.get_L_curve()

    ax3.scatter(rho, eta)
    ax3.scatter(rho[alpha_idx], eta[alpha_idx], c='r', facecolor=None)
    plt.show()

    print(spc.alpha)
