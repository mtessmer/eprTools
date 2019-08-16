import numpy as np
from scipy.optimize import minimize, curve_fit, nnls
from scipy.interpolate import interp1d
from scipy.special import fresnel
from sklearn.linear_model.base import LinearModel
from eprTools.tntnn import tntnn
import matplotlib.pyplot as plt
from time import time
import cvxopt as cvo
from numba import njit, cuda


class DEERSpec:
    """
    Double elctron-electron resonance spectrum object created from a dipolar evolution trace.

    ----------
    time : ndarray
        Time points of experimental trace in nanoseconds.
    spec_real : ndarray
        Real values of experimental DEER trace
    spec_imag : ndarray
        Imaginary values of experimental DEER trace
    r_min : float, default 15.0
        Minimum distance value in angstroms to include in kernel for fitting P(r). Distance less than r_min will not
        be calculated as part of P(r). r_min should not be below 15 angstroms.
    r_max : float, default 80.0
        Maximum distance value in angstroms to include in kernel for fitting P(r). Note that large ranges of the kernel
        increase the condition number making it harder to fit. Additionally large distances can easily be artifacts of
        background correction and should evaluated critically with respect to the length of the time trace.
    do_phase : boolean, default True
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

    def __init__(self, time, spec_real, spec_imag, r_min, r_max, do_phase):

        # Working values
        self.time = time
        self.real = spec_real
        self.imag = spec_imag

        # Tikhonov fit results
        self.fit = None
        self.alpha = None
        self.alpha_idx = None
        self.rho = None
        self.eta = None
        self.P = None
        self.fit_time = None

        # Kernel parameters
        self.r_min = r_min
        self.r_max = r_max
        self.kernel_len = 216
        self.r = np.linspace(r_min, r_max, self.kernel_len)

        # Default phase and trimming parameters
        self.phi = None
        self.do_phase = do_phase
        self.trim_length = None
        self.zt = None
        self.L_criteria = 'gcv'

        # Default background correction parameters
        self.background_kind = '3D'
        self.background_k = 1
        self.background_fit_t = None

        # Background correction results
        self.dipolar_evolution = None
        self.background = None
        self.background_param = None

        # Raw Data Untouched
        self.raw_time = time
        self.raw_spec_real = spec_real
        self.raw_spec_imag = spec_imag

        self._update()

    @classmethod
    def from_file(cls, file_name, r_min=15, r_max=80):

        with open(file_name, 'rb') as file:
            spec = np.frombuffer(file.read(), dtype='>d')

        # Look for time from DSC file
        param_file = file_name[:-3] + 'DSC'
        param_dict = {}
        try:
            with open(param_file, 'r') as file:
                for line in file:

                    # Skip blank lines and lines with comment chars
                    if line.startswith(("*", "#", "\n")):
                        continue

                    # Add keywords to param_dict
                    else:
                        line = line.split()
                        try:
                            key = line[0]
                            val = [arg.strip() for arg in line[1:]]
                        except IndexError:
                            key = line
                            val = None
                        param_dict[key] = val
        except OSError:
            print("Error: No parameter file found")

        # Don't perform phase correction if experiment was not phase cycled
        do_phase = True
        if param_dict['PlsSPELLISTSlct'][0] == 'none':
            do_phase = False

        # Calculate time axis data from experimental params
        points = int(param_dict['XPTS'][0])
        time_min = float(param_dict['XMIN'][0])
        time_width = float(param_dict['XWID'][0])

        time_max = time_min + time_width
        time = np.linspace(time_min, time_max, points)

        # Separate real and imaginary components of experimental spectrum
        spec_real = spec[0::2]
        spec_imag = spec[1::2]

        # Construct DEERSpec object
        return cls(time, spec_real, spec_imag, r_min, r_max, do_phase)

    @classmethod
    def from_array(cls, time, spec_real, spec_imag,  r_min=15, r_max=80):

        # Create 0 vector for imaginary component if it is not provided
        if not spec_imag:
            spec_imag = np.zeros(len(spec_real))

        # Construct DEERSpec object
        return cls(time, spec_real, spec_imag, r_min, r_max, do_phase=False)

    def _update(self):
        """
        Update all self variables except fit. internal use only.
        """

        self.trim()
        self.zero_time()

        if self.do_phase:
            self.phase()

        self.correct_background()
        self.compute_kernel()

    def set_kernel_len(self, length=250):
        """
        Change the size of the kernel. Does not change the range of time or distance

        length : int, default=250
            length of kernel array. Note that large kernels require more time to fit.

        See Also
        --------
        set_kernel_r

        Examples
        --------

        """
        self.kernel_len = length
        self.r = np.linspace(self.r_min, self.r_max, self.kernel_len)
        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)
        self._update()

    def get_L_curve(self, length=80, set_alpha=False):
        """
        returns L-curve and optimal alpha index

        length: int, default 80
            number of alpha values to test when calculating L-curve
        set_alpha: boolean, default False
            set object attribute self.alpha after calculating L-curve and determining optimal alpha. Should only be
            required by get_fit()

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
        if self.alpha_idx:
            return self.rho, self.eta, self.alpha_idx

        else:
            # If alpha is already defined construct L-curve centered around it
            if self.alpha:
                log_min = np.logspace(np.log10(self.alpha) - 4, np.log10(self.alpha), np.floor(length / 2))
                log_max = np.logspace(np.log10(self.alpha), np.log10(self.alpha) + 4, np.ceil(length / 2))
                alpha_list = np.concatenate([log_min, log_max])
            # Else center L-curve at 1
            # TODO: implement dynamic choice of L-curve range to account for dynamic kernels
            else:
                alpha_list = np.logspace(-4, 4, length)

            # Preallocate L-Curve data
            rho = np.zeros(len(alpha_list))
            eta = np.zeros(len(alpha_list))
            best_alpha_score = 100000
            best_alpha = 1

            # Compute (gcv or aic) score of each alpha and store the optimal alpha and score
            for i, alpha in enumerate(alpha_list):
                P, temp_fit = self.get_P(alpha)
                Serr = (self.y + self.y_offset) - temp_fit
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

    def set_kernel_r(self, r_min=15, r_max=80):
        """
        Set the distance range of the kernel
        r_min: int, float, default 15
            minimum distance value of kernel in angstroms
        r_max: int, float, default 80
            maximum distance value of kernel in angstroms

        See Also
        --------
        set_kernel_len

        Examples
        --------
        >>> from eprTools import DEERSpec
        >>>

        """
        self.r = np.linspace(r_min, r_max, self.kernel_len)
        self.r_min = r_min
        self.r_max = r_max
        self._update()

    def set_phase(self, phi=0, degrees=True):

        if not degrees:
            self.phi = phi
        elif degrees:
            self.phi = phi * np.pi / 180.0

        self._update()

    def set_trim(self, trim=None):

        self.trim_length = trim
        self._update()

    def set_zero_time(self, zt=None):

        self.zt = zt;
        self._update()

    def set_background_correction(self, kind='3D', k=1, fit_time=None):

        self.background_kind = kind
        self.background_k = k
        self.background_fit_t = fit_time
        self._update()

    def set_L_criteria(self, mode):

        self.L_criteria = mode
        self._update()

    def compute_kernel(self):

        # Compute Kernel
        omega_dd = (2 * np.pi * 52.0410) / (self.r ** 3)
        trig_term = np.outer(self.fit_time, omega_dd)
        z = np.sqrt((6 * trig_term / np.pi))
        S_z, C_z = fresnel(z)
        SzNorm = S_z / z
        CzNorm = C_z / z

        cos_term = np.cos(trig_term)
        sin_term = np.sin(trig_term)
        K = CzNorm * cos_term + SzNorm * sin_term

        # Define L matrix
        L = np.zeros((self.kernel_len - 2, self.kernel_len))
        spots = np.arange(self.kernel_len - 2)
        L[spots, spots] = 1
        L[spots, spots + 1] = - 2
        L[spots, spots + 2] = 1
        self.L = L

        # Compress Data to kernel dimensions
        f = interp1d(self.time, self.dipolar_evolution)
        kData = f(self.fit_time)

        # Preprocess kernel and data for nnls fitting
        self.K, self.y, X_offset, self.y_offset, X_scale = LinearModel._preprocess_data(
            K, kData, True, False, True, sample_weight=None)
        self.L = L

    def trim(self):

        self.real = self.raw_spec_real
        self.imag = self.raw_spec_imag
        self.time = self.raw_time

        # normalize working values
        if min(self.real < 0):
            self.real = self.real - 2 * min(self.real)

            self.imag = self.imag / max(self.real)
            self.real = self.real / max(self.real)

        if not self.trim_length:

            # take last quarter of data
            start = -int(len(self.real) / 3)
            window = 11

            # get minimum std
            min_std = self.real[-window:].std()
            min_i = -window
            for i in range(start, -window):
                test_std = self.real[i:i + window].std()
                if test_std < min_std:
                    min_std = test_std
                    min_i = i

            max_std = 3 * min_std
            cutoff = len(self.real)

            for i in range(start, - window):
                test_std = self.real[i:i + window].std()
                if (test_std > max_std) & (i > min_i):
                    cutoff = i
                    break

        elif self.trim_length:
            cutoff = self.trim_length
            f_spec_real = interp1d(self.time, self.real, 'cubic')
            f_spec_imag = interp1d(self.time, self.imag, 'cubic')

            self.time = np.arange(self.time.min(), self.time.max())
            self.real = f_spec_real(self.time)
            self.imag = f_spec_imag(self.time)

        self.time = self.time[:cutoff]
        self.real = self.real[:cutoff]
        self.imag = self.imag[:cutoff]

    def phase(self):

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

        complex_data = complex_data * np.exp(1j * self.phi)
        self.real = np.real(complex_data) / np.real(complex_data).max()
        self.imag = np.imag(complex_data) / np.real(complex_data).max()

    def zero_time(self):

        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)

            if len(xData) != len(data):
                xData = xData[:-1]

            return np.dot(data, xData)

        # Interpolate data
        f_spec_real = interp1d(self.time, self.real, 'cubic')
        f_spec_imag = interp1d(self.time, self.imag, 'cubic')

        self.time = np.arange(self.time.min(), self.time.max())
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
            for i in range(half_spec_max_idx, 500):
                lFrame = i - half_spec_max_idx
                uFrame = i + half_spec_max_idx + 1

                test_moment = zero_moment(self.real[lFrame: uFrame])

                if abs(test_moment) < abs(low_moment):
                    low_moment = test_moment
                    spec_max_idx = i

        else:
            spec_max_idx = self.zt

        # Adjust time to appropriate zero
        self.time = self.time - self.time[spec_max_idx]

        # Remove time < 0
        self.time = self.time[spec_max_idx:]
        self.real = self.real[spec_max_idx:]
        self.imag = self.imag[spec_max_idx:]

        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)

    def correct_background(self):

        # calculate t0 for fit_t if none given
        if not self.background_fit_t:
            self.background_fit_t = (int(len(self.time) / 8))

        # Use last 3/4 of data to fit background
        fit_time = self.time[self.background_fit_t:]
        fit_real = self.real[self.background_fit_t:]

        if self.background_kind in ['3D', '2D']:
            if self.background_kind == '2D':
                d = 2
            elif self.background_kind == '3D':
                d = 3

            def homogeneous_3d(t, a, k, j):
                return a * np.exp(-k * (t ** (d / 3)) + j)

            popt, pcov = curve_fit(homogeneous_3d, fit_time, fit_real, p0=(1, 1e-5, 1e-2))

            self.background = homogeneous_3d(self.time, *popt)
            self.background_param = popt

            self.dipolar_evolution = self.real - homogeneous_3d(self.time, *popt) + (popt[0] * np.exp(popt[2]))

        elif self.background_kind == 'poly':

            popt = np.polyfit(fit_time, fit_real, deg=self.background_k)
            self.background = np.polyval(popt, self.time)
            self.background_param = popt
            self.dipolar_evolution = self.real - np.polyval(popt, self.time) + popt[-1]


    def get_fit(self, alpha=None, true_min=False, fit_method='nnls'):

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

        if self.fit_method== 'nnls':
            C = np.concatenate([self.K, alpha * self.L])
            d = np.concatenate([self.y, np.zeros(shape=self.kernel_len - 2)])

            P, _ = nnls(C, d)

        elif self.fit_method == 'tntnn':
            C = np.concatenate([self.K, alpha * self.L])
            d = np.concatenate([self.y, np.zeros(shape=self.kernel_len - 2)])

            P, _ = tntnn(C, d, use_AA=True)

        elif self.fit_method == 'nnls_gd':
            XTX = self.K.T.dot(self.K) + (alpha**2)*self.L.T.dot(self.L)
            XTY = self.K.T.dot(self.y)

            P = NNLS_GD(XTX, XTY)

        elif self.fit_method == 'cvx':
            P = self.get_P_cvx(alpha)

        else:
            raise NameError('Fit method {} is not supported'.format(self.fit_method))
        fit = self.K.dot(P) + self.y_offset
        return P, fit

    def get_P_cvx(self, alpha):

        # non-negative solution to get a non-negative P -- adapted from Stephan Rein's GloPel
        K = self.K
        L = self.L
        points = len(K)

        # Get initial matrices of optimization
        pre_result = (K.T.dot(K) + alpha * L.T.dot(L))

        P = np.linalg.inv(pre_result).dot(K.T).dot(self.y)
        P = P.clip(min=0)

        B = cvo.matrix(pre_result)
        A = cvo.matrix(-(K.T.dot(self.y.T)))

        # Minimize with CVXOPT constrained to > 0
        lower_bound = cvo.matrix(np.zeros(points))
        G = -cvo.matrix(np.eye(points, points))
        cvo.solvers.options['show_progress'] = False
        P = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))['x']
        P = np.asarray(P).reshape(points, )

        return P

    def get_AIC_score(self, alpha, fit):

        s_error = (self.y + self.y_offset) - fit
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)

        nt = self.kernel_len
        score = nt * np.log((np.linalg.norm(s_error) ** 2) / nt) + (2 * np.trace(H_alpha))

        return score

    def get_score(self, alpha, fit = None):

        if fit is None:
            P, fit = self.get_P(alpha)

        if self.L_criteria == 'gcv':
            return self.get_GCV_score(alpha, fit)
        elif self.L_criteria == 'aic':
            return self.get_AIC_score(alpha, fit)

    def get_GCV_score(self, alpha, fit):

        s_error = (self.y + self.y_offset) - fit
        K_alpha = np.linalg.inv((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L))).dot(self.K.T)

        H_alpha = self.K.dot(K_alpha)
        nt = self.kernel_len
        score = np.linalg.norm(s_error) ** 2 / (1 - np.trace(H_alpha) / nt) ** 2
        return score


#import cProfile
#cp = cProfile.Profile()


def do_it_for_me(filename, true_min=False, fit_method='nnls'):
    #cp.enable()
    t1 = time()
    spc = DEERSpec.from_file(filename)
    spc.get_fit(true_min=true_min, fit_method=fit_method)
    t2 = time()
    print(spc.background_param)
    #cp.disable()

    #cp.print_stats()
    print("Fit computed in {}".format(t2 - t1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10.5])
    ax1.plot(spc.time, spc.dipolar_evolution)
    ax1.plot(spc.fit_time, spc.fit)
    ax2.plot(spc.r, spc.P)
    plt.show()

    # Get L-curve
    t1 = time()
    rho, eta, alpha_idx = spc.get_L_curve()
    t2 = time()
    print("L-curve computed in {}".format(t2 - t1))

    fig2, ax = plt.subplots()
    ax.scatter(rho, eta)
    ax.scatter(rho[alpha_idx], eta[alpha_idx], c='r', facecolor=None)
    plt.show()

    print(spc.alpha)

@njit
def solveNQP(Q, q, epsilon, max_n_iter):
    # Initialize
    n_cols = len(q)
    index = np.arange(n_cols)
    x = np.zeros(n_cols)
    x_diff = np.zeros(n_cols)
    grad_f = q.copy()
    grad_f_bar = q.copy()
    

    # Loop over iterations  
    for i in range(max_n_iter):

        # Get passive set information
        passive_set = np.logical_or(x > 0, grad_f < 0)

        # Calculate gradient
        grad_f_bar[:] = grad_f
        grad_f_bar[~passive_set] = 0
        grad_norm = np.vdot(grad_f_bar, grad_f_bar)

        # Abort?
        if (~passive_set.any() or grad_norm < epsilon):
            break

        # Exact line search
        alpha_den = np.vdot(grad_f_bar, Q.dot(grad_f_bar))
        alpha = grad_norm / alpha_den

        # Update x
        x_diff = -x.copy()
        x -= alpha * grad_f_bar
        x = np.maximum(x, 0.)
        x_diff += x

        # Update gradient
        grad_f += Q.dot(x_diff)

    # Return
    return x
#TODO Prcompile NNLS algorithm
#@njit
def NNLS_GD(XTX, XTY, epsilon=1e-8):
    max_n_iter = 5 * XTX.shape[1]

    # Normalization factors
    H_diag = np.diag(XTX).copy().flatten()
    Q_den  = np.sqrt(np.outer(H_diag, H_diag))
    q_den  = np.sqrt(H_diag)

    # Normalize
    Q =  XTX / Q_den
    q = -XTY / q_den
    # Solve NQP
    y = solveNQP(Q, q, epsilon, max_n_iter)

    # Undo normalization
    x = y / q_den

    # Return
    return x.T

@njit
def NNLS_CD(XTX, XTY, epsilon=1e-8):
    max_n_iter = 5 * XTX.shape[1]

    # Initialize
    x = np.zeros((np.shape(XTX)[1]))
    f = -XTY
    mu = f.copy()

    # Loop over iterations
    for i in range(max_n_iter):

        # Abort?
        Hxf = XTX.dot(x) + f
        criterion_1 = np.all(Hxf >= -epsilon)
        criterion_2 = np.all(Hxf[x > 0] <= epsilon)
        if (criterion_1 and criterion_2):
            break

        # Loop over coordinates
        for k in range(np.shape(XTX)[1]):
            x_diff = -x[k]
            x[k] = np.maximum(0., x[k] - mu[k] / XTX[k, k])
            x_diff += x[k]
            mu += x_diff * XTX[:, k]

    # Return
    return x
