import numpy as np
from scipy.optimize import minimize, curve_fit, nnls
from scipy.interpolate import interp1d
from scipy.special import fresnel
from sklearn.linear_model.base import LinearModel
from eprTools.tntnn import tntnn
import matplotlib.pyplot as plt
from time import time


def do_it_for_me(filename):
    t1 = time()
    spc = DEERSpec.from_file(filename)
    spc.get_fit()
    t2 = time()
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


class DEERSpec:

    def __init__(self, time, yreal, yimag, rmin, rmax, do_phase):

        # Working values
        self.time = time
        self.real = yreal
        self.imag = yimag

        # Tikhonov fit results
        self.fit = None
        self.alpha = None
        self.alpha_idx = None
        self.rho = None
        self.eta = None
        self.P = None
        self.fit_time = None

        # Kernel parameters
        self.r_min = rmin
        self.r_max = rmax
        self.kernel_len = 256
        self.r = np.linspace(rmin, rmax, self.kernel_len)

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
        self.raw_real = yreal
        self.raw_imag = yimag

        self.update()

    @classmethod
    def from_file(cls, file_name, r_min=15, r_max=80):
        d = {}
        ydata = []
        with open(file_name, 'rb') as file:

            if file_name[-3:] == 'DTA':
                ydata = np.frombuffer(file.read(), dtype='>d')

                # Look for Xdata from DSC file
                paramfile = file_name[:-3] + 'DSC'
                try:
                    with open(paramfile, 'r') as f2:
                        for line in f2:
                            # Skip blank lines and lines with comment chars
                            if line.startswith(("*", "#", "\n")):
                                continue
                            else:
                                line = line.split()
                                try:
                                    key = line[0]
                                    val = [arg.strip() for arg in line[1:]]
                                except IndexError:
                                    key = line
                                    val = None
                                d[key] = val
                except OSError:
                    print("Error: No parameter file found")

        do_phase = True
        if d['PlsSPELLISTSlct'][0] == 'none':
            do_phase = False

        xpoints = int(d['XPTS'][0])
        xmin = float(d['XMIN'][0])
        xwid = float(d['XWID'][0])

        xmax = xmin + xwid
        xdata = np.linspace(xmin, xmax, xpoints)

        yreal = ydata[::2]
        yimag = ydata[1::2]

        return cls(xdata, yreal, yimag, r_min, r_max, do_phase)

    def update(self):

        self.trim()
        self.zero_time()

        if self.do_phase:
            self.phase()

        self.correct_background()
        self.compute_kernel()

    def set_kernel_len(self, length):
        self.kernel_len = length
        self.r = np.linspace(self.r_min, self.r_max, self.kernel_len)
        self.fit_time = np.linspace(1e-6, self.time.max(), self.kernel_len)
        self.update()

    def get_L_curve(self, length=20, set_alpha=False):

        if self.alpha_idx:
            return self.rho, self.eta, self.alpha_idx

        else:
            alpha_list = np.logspace(-4, 4, length)
            rho = np.zeros(len(alpha_list))
            eta = np.zeros(len(alpha_list))
            best_alpha_score = 100000
            best_alpha = 1

            # TODO: paralellize
            for i, alpha in enumerate(alpha_list):
                P, temp_fit = self.get_P(alpha)
                Serr = (self.y + self.y_offset) - temp_fit
                rho[i] = np.log(np.linalg.norm(Serr))
                eta[i] = np.log(np.linalg.norm((np.dot(self.L, P))))
                temp_score = self.get_score(alpha)
                if temp_score < best_alpha_score:
                    best_alpha_score = temp_score
                    best_alpha = alpha

            if set_alpha:
                self.alpha = best_alpha

            difference = np.abs(alpha_list - self.alpha)

            self.alpha_idx = np.argmin(difference)
            self.rho = rho
            self.eta = eta

            return self.rho, self.eta, self.alpha_idx

    def set_kernel_r(self, rmin=15, rmax=80):
        self.r = np.linspace(rmin, rmax, self.kernel_len)
        self.r_min = rmin
        self.r_max = rmax
        self.update()

    def set_phase(self, phi=0, degrees=True):

        if not degrees:
            self.phi = phi
        elif degrees:
            self.phi = phi * np.pi / 180.0

        self.update()

    def set_trim(self, trim=None):
        self.trim_length = trim
        self.update()

    def set_zero_time(self, zt=None):
        self.zt = zt;
        self.update()

    def set_background_correction(self, kind='3D', k=1, fit_time=None):
        self.background_kind = kind
        self.background_k = k
        self.background_fit_t = fit_time
        self.update()

    def set_L_criteria(self, mode):
        self.L_criteria = mode
        self.update()

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
        self.real = self.raw_real
        self.imag = self.raw_imag
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
            freal = interp1d(self.time, self.real, 'cubic')
            fimag = interp1d(self.time, self.imag, 'cubic')

            self.time = np.arange(self.time.min(), self.time.max())
            self.real = freal(self.time)
            self.imag = fimag(self.time)

        self.time = self.time[:cutoff]
        self.real = self.real[:cutoff]
        self.imag = self.imag[:cutoff]

    def phase(self):
        # Make complex array for phase adjustment
        cData = self.real + 1j * self.imag

        if self.phi is None:
            # Initial guess for phase shift
            phi0 = np.arctan2(self.imag[-1], self.real[-1])

            # Use last 7/8ths of data to fit phase
            fit_set = cData[int(round(len(cData) / 8)):]

            def get_imag_norm_squared(phi):
                temp = np.imag(fit_set * np.exp(1j * phi))
                return np.dot(temp, temp)

            # Find Phi that minimizes norm of imaginary data
            phi = minimize(get_imag_norm_squared, phi0)
            phi = phi.x
            temp = cData * np.exp(1j * phi)

            # Test for 180 degree inversion of real data
            if np.real(temp).sum() < 0:
                phi = phi + np.pi

            self.phi = phi

        cData = cData * np.exp(1j * self.phi)
        self.real = np.real(cData) / np.real(cData).max()
        self.imag = np.imag(cData) / np.real(cData).max()

    def zero_time(self):

        def zero_moment(data):
            xSize = int(len(data) / 2)
            xData = np.arange(-xSize, xSize + 1)

            if len(xData) != len(data):
                xData = xData[:-1]

            return np.dot(data, xData)

        # Interpolate data
        freal = interp1d(self.time, self.real, 'cubic')
        fimag = interp1d(self.time, self.imag, 'cubic')

        self.time = np.arange(self.time.min(), self.time.max())
        self.real = freal(self.time)
        self.imag = fimag(self.time)

        if not self.zt:
            # ake zero_moment of all windows tx(tmax)/2 and find minimum
            tmax = self.real.argmax()
            half_tmax = int(tmax / 2)

            lFrame = tmax - half_tmax
            uFrame = tmax + half_tmax + 1
            low_moment = zero_moment(self.real[lFrame: uFrame])

            # Only look in first 500ns of data
            for i in range(half_tmax, 500):
                lFrame = i - half_tmax
                uFrame = i + half_tmax + 1

                test_moment = zero_moment(self.real[lFrame: uFrame])

                if abs(test_moment) < abs(low_moment):
                    low_moment = test_moment
                    tmax = i

        else:
            tmax = self.zt

        # Adjust time to appropriate zero
        self.time = self.time - self.time[tmax]

        # Remove time < 0
        self.time = self.time[tmax:]
        self.real = self.real[tmax:]
        self.imag = self.imag[tmax:]

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

    def get_fit(self, alpha=None, truemin=False):

        if alpha is None and truemin:

            res = minimize(self.get_score, 1, method='Nelder-Mead')
            self.alpha = res.x

        elif alpha is None:
            rho, eta, alpha_idx = self.get_L_curve(set_alpha=True)
        else:
            self.alpha = alpha

        P, self.fit = self.get_P(self.alpha)
        self.P = P / np.sum(P)

    def get_P(self, alpha):

        C = np.concatenate([self.K, alpha * self.L])
        d = np.concatenate([self.y, np.zeros(shape=self.kernel_len - 2)])

        if self.kernel_len > 1024:
            P = tntnn(C, d, use_AA=True)
        else:
            # start = time()
            # P = nnlsm_blockpivot(C,d)
            # print('nnls_bpp:', time() - start)
            # start = time()
            P = nnls(C, d)
            # print('nnls:', time() - start)

        temp_fit = self.K.dot(P[0]) + self.y_offset
        return P[0], temp_fit

    def get_P_cvex(self, alpha):
        # non-negative solution to get a non-negative P -- adapted from Stephan Rein's GloPel
        K = self.K
        L = self.L
        points = len(K)

        # Get initial matrices of optimization
        preresult = (K.T.dot(K) + alpha * L.T.dot(L))

        P = np.linalg.inv(preresult).dot(K.T).dot(self.y)
        P = P.clip(min=0)

        B = cvo.matrix(preresult)
        A = cvo.matrix(-(K.T.dot(self.y.T)))

        # Minimize with CVXOPT constrained to > 0
        lower_bound = cvo.matrix(np.zeros(points))
        G = -cvo.matrix(np.eye(points, points))
        cvo.solvers.options['show_progress'] = False
        Z = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))['x']
        Z = np.asarray(Z).reshape(points, )
        temp_fit = K.dot(Z) + self.y_offset
        return Z, temp_fit

    def get_AIC_score(self, alpha):
        P, temp_fit = self.get_P(alpha)
        Serr = (self.y + self.y_offset) - temp_fit
        K_alpha, _, _, _ = np.linalg.lstsq((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L)), self.K.T,
                                           rcond=None)
        H_alpha = self.K.dot(K_alpha)

        nt = self.kernel_len
        score = nt * np.log((np.linalg.norm(Serr) ** 2) / nt) + (2 * np.trace(H_alpha))

        return score

    def get_score(self, alpha):
        if self.L_criteria == 'gcv':
            return self.get_GCV_score(alpha)
        elif self.L_criteria == 'aic':
            return self.get_AIC_score(alpha)

    def get_GCV_score(self, alpha):

        P, temp_fit = self.get_P(alpha)

        Serr = (self.y + self.y_offset) - temp_fit
        K_alpha, _, _, _ = np.linalg.lstsq((self.K.T.dot(self.K) + (alpha ** 2) * self.L.T.dot(self.L)), self.K.T,
                                           rcond=None)
        H_alpha = self.K.dot(K_alpha)
        nt = self.kernel_len
        score = np.linalg.norm(Serr) ** 2 / (1 - np.trace(H_alpha) / nt) ** 2
        return score
