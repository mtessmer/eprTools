import numbers
import warnings
from collections.abc import Sized
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy.constants import physical_constants, mu_0, h, hbar

ge, ge_unit, ge_uncertainty = physical_constants['electron g factor']
muB, muB_unit, muB_uncertainty = physical_constants['Bohr magneton']
D = (mu_0/4/np.pi)*(muB*ge)**2/hbar

def generate_kernel(r=(15, 80), time=3500, **kwargs):
    """
    Generate DEER kernel using  angstroms and nanoseconds

    :param r: float, tuple, sized container
        Distance domain of the kernel.
        if float, distance domain will go from 1-float
        if tuple of length 2, the distance domain represents the min and max

    :param time: float
        maximum time covered by kernel (ns)

    :param size: int
        axis size of kernel matrix

    :return: numpy ndarray
        DEER Kernel matrix
    """
    size = kwargs.get('size', 256)
    r = setup_r(r, size=size)

    # Interpret time domain
    if isinstance(time, numbers.Number):
        time = np.linspace(0, time, size)

    elif isinstance(time, Sized):
        if len(time) == 2:
            time = np.linspace(time[0], time[1], size)
        else:
            time = np.asarray(time)

    else:
        raise ValueError("Unrecognized value for time domain.")

    g = kwargs.get('g', np.array([ge, ge]))
    omega_dd = (mu_0/2)*muB**2*g[0]*g[1]/h*1e21/(r**3)  # rad μs^-1
    trigterm = np.outer(np.abs(time), omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_z, C_z = fresnel(z) / z

    K = C_z * np.cos(trigterm) + S_z * np.sin(trigterm)

    # Correct for error introduced by divide by zero error
    K[time == 0] = 1.

    return K, r, time


def read_param_file(param_file):
    if param_file is None:
        return {}

    param_dict = {'DESC': {},
                  'SPL': {},
                  'DSL': {}}
    active_dict = param_dict['DESC']
    with (open(param_file, 'r', encoding='CP1252') as file):

        for line in file:
            # Skip blank lines and lines with comment chars
            if line.startswith(("*", "\n", ' ')):
                continue

            elif line.startswith('#'):
                section = line.split(maxsplit=1)[0][1:]
                active_dict = param_dict[section]
                continue


            elif line.startswith('.DVC'):
                new_section = line.split()[1].strip(',')
                param_dict['DSL'][new_section] = {}
                active_dict = param_dict['DSL'][new_section]
                continue

            # Add keywords to param_dict
            kv = line.split(maxsplit=1)
            key, val = kv if len(kv) == 2 else (kv[0], '')

            # Parse string data
            if val.startswith("'"):
                val = val.strip("'")

            # Parse array data
            val = val.strip()
            if val.startswith('{'):
                end_idx = val.find('}')
                if end_idx == -1:
                    continue



                header, vals = val[1:end_idx], val[end_idx+2:]
                dim, shape, extra = header.split(';')
                shape = tuple(int(s) for s in shape.split(','))
                vals = np.array([float(x) for x in vals.split(',')])
                vals.shape = shape
                val = np.squeeze(vals)


            # append multiline values
            if isinstance(val, str):
                while val.endswith('\\'):
                    val = val[:-1]
                    val += next(file).strip()

                val = val.replace('\\n', '\n')

            active_dict[key] = val

    return param_dict


def read_bruker(data_file, return_params=False, phase=False):
    param_file = data_file[:-3] + 'DSC'

    try:
        param_dict = read_param_file(param_file)
    except OSError:
        print("Warning: No parameter file found")

    desc = param_dict['DESC']

    # Calculate time axis data from experimental params
    points = int(desc['XPTS'])
    time_min = float(desc['XMIN'])
    time_width = float(desc['XWID'])

    if 'YPTS' in desc.keys():
        n_scans = int(desc['YPTS'])
    else:
        n_scans = 1

    time_max = time_min + time_width
    time = np.linspace(time_min, time_max, points)

    # Read data
    signal = np.fromfile(data_file, dtype='>d')

    # Reshape and form complex array
    signal.shape = (-1, 2)
    signal = signal[:, 0] + 1j * signal[:, 1]

    # If the experiment is 2D
    if n_scans > 1:
        # Reshape to a 2D matrix
        signal.shape = (n_scans, -1)

        # Trim scans cut short of the predetermined number
        mask = ~np.all(signal == 0, axis=1)
        if ~mask.sum():
            signal = signal[mask]   # All scans that are 0
            signal = signal[:-1]    # Last scan that wasn't all 0 probably was cut short

    if phase:
        signal = opt_phase(signal)
    if return_params:
        return time, signal, param_dict
    else:
        return time, signal


def reg_operator(r, kind='L2'):
    loffset, uoffset = (0, None) if kind[-1] == '+' else (1, -1)

    L = np.zeros((len(r), len(r)))
    diag = np.arange(len(r))

    L[diag[:-1], diag[1:]] = 1
    L[diag, diag] = - 2
    L[diag[1:], diag[:-1]] = 1

    return L[loffset:uoffset]

def hccm(J, *args):
    """
    Heteroscedasticity Consistent Covariance Matrix (HCCM)
    ======================================================
    Computes the heteroscedasticity consistent covariance matrix (HCCM) of
    a given LSQ problem given by the Jacobian matrix (J) and the covariance
    matrix of the data (V). If the residual (res) is specified, the
    covariance matrix is estimated using some of the methods specified in
    (mode). The HCCM are valid for both heteroscedasticit and
    homoscedasticit residual vectors.
    Usage:
    ------
    C = hccm(J,V)
    C = hccm(J,res,mode)
    Arguments:
    ----------
    J (NxM-element array)
        Jacobian matrix of the residual vector
    res (N-element array)
        Vector of residuals
    mode (string)
        HCCM estimator, options are:
            'HC0' - White, H. (1980)
            'HC1' - MacKinnon and White, (1985)
            'HC2' - MacKinnon and White, (1985)
            'HC3' - Davidson and MacKinnon, (1993)
            'HC4' - Cribari-Neto, (2004)
            'HC5' - Cribari-Neto, (2007)
    Returns:
    --------
    C (MxM-element array)
       Heteroscedasticity consistent covariance matrix
    References:
    ------------
    [1]
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838
    DOI: 10.2307/1912934
    [2]
    MacKinnon and White, (1985). Some heteroskedasticity-consistent covariance
    matrix estimators with improved finite sample properties. Journal of Econometrics, 29 (1985),
    pp. 305-325. DOI: 10.1016/0304-4076(85)90158-7
    [3]
    Davidson and MacKinnon, (1993). Estimation and Inference in Econometrics
    Oxford University Press, New York.
    [4]
    Cribari-Neto, F. (2004). Asymptotic inference under heteroskedasticity of
    unknown form. Computational Statistics & Data Analysis, 45(1), 215-233
    DOI: 10.1016/s0167-9473(02)00366-3
    [5]
    Cribari-Neto, F., Souza, T. C., & Vasconcellos, K. L. P. (2007). Inference
    under heteroskedasticity and leveraged data. Communications in Statistics –
    Theory and Methods, 36(10), 1877-1888. DOI: 10.1080/03610920601126589
    """

    # Unpack inputs
    if len(args) == 2:
        res, mode = args
        V = []
    elif len(args) == 1:
        V = args[0]

    # Hat matrix
    H = J @ np.linalg.pinv(J.T @ J) @ J.T
    # Get leverage
    h = np.diag(H)
    # Number of parameters (k) & Number of variables (n)
    n, k = np.shape(J)

    if not V:
        # Select estimation method using established nomenclature
        if mode.upper() == 'HC0':  # White,(1980),[1]
            # Estimate the data covariance matrix
            V = np.diag(res ** 2)

        elif mode.upper() == 'HC1':  # MacKinnon and White,(1985),[2]
            # Estimate the data covariance matrix
            V = n / (n - k) * np.diag(res ** 2)

        elif mode.upper() == 'HC2':  # MacKinnon and White,(1985),[2]
            # Estimate the data covariance matrix
            V = np.diag(res ** 2 / (1 - h))

        elif mode.upper() == 'HC3':  # Davidson and MacKinnon,(1993),[3]
            # Estimate the data covariance matrix
            V = np.diag(res / (1 - h)) ** 2

        elif mode.upper() == 'HC4':  # Cribari-Neto,(2004),[4]
            # Compute discount factor
            delta = min(4, n * h / k)
            # Estimate the data covariance matrix
            V = np.diag(res ** 2. / ((1 - h) ** delta))

        elif mode.upper() == 'HC5':  # Cribari-Neto,(2007),[5]
            # Compute inflation factor
            k = 0.7
            alpha = min(max(4, k * max(h) / np.mean(h)), h / np.mean(h))
            # Estimate the data covariance matrix
            V = np.diag(res ** 2. / (np.sqrt((1 - h) ** alpha)))

        else:
            raise KeyError('HCCM estimation mode not found.')

    # Heteroscedasticity Consistent Covariance Matrix (HCCM) estimator
    C = np.linalg.pinv(J.T @ J) @ J.T @ V @ J @ np.linalg.pinv(J.T @ J)
    return C

def get_imag_norms_squared(phi, V):
    V_imag = np.imag(V[:, None] * np.exp(1j * phi)[None, :, None])

    return (V_imag * V_imag).sum(-1)


def opt_phase(V, return_params=False):
    V = np.atleast_2d(V)

    # Calculate 3 points of cost function which should be a smooth continuous sine wave
    phis = np.array([0, np.pi / 2, np.pi]) / 2
    costs = get_imag_norms_squared(phis, V)

    # Calculate sine function fitting 3 points
    offset = (costs[:, 0] + costs[:, 2]) / 2
    phase_shift = np.arctan2(costs[:, 0] - offset, costs[:, 1] - offset)

    # Calculate phi by calculating the phase when the derivative of the sine function is 0 and using the second
    # derivative to ensure it is a minima and not a maxima
    possible_phis = np.array([(np.pi / 2 - phase_shift) / 2, (3 * np.pi / 2 - phase_shift) / 2]).T
    second_deriv = -np.sin(2 * possible_phis + phase_shift[:, None])
    opt_phase = possible_phis[second_deriv > 0]

    # Check to ensure the real component is positive
    temp_V = V * np.exp(1j * opt_phase)[:, None]
    opt_phase[temp_V.sum(axis=1) < 0] += np.pi
    V = V * np.exp(1j * opt_phase)[:, None]

    if return_params:
        return np.squeeze(V), np.squeeze(opt_phase)
    else:
        return np.squeeze(V)

def fit_zero_time(raw_time, raw_real, return_params=False, return_fit=False):
    """
    Obtain the zero time and amplitude by fitting the first part of the trace to a 5th order polynomial

    :param raw_time: np.ndarray
        The raw experimental time points
    :param raw_real:
        The raw real component of the experimental  data

    :return fit_time, normal_V:
        the time and amplitude adjusted time and echo data
    """

    # Get approximate location of zero time
    idxmax = np.argmax(raw_real)
    idxmax = np.maximum(idxmax, 10)

    # Fit data surrounding approximate zero-time to 5th order polynomial
    fit = np.polyfit(raw_time[:2 * idxmax], raw_real[: 2 * idxmax], 7)

    # Interpolate to find t0
    dense_time = np.linspace(raw_time[0], raw_time[2 * idxmax], 256)
    dense_V = np.polyval(fit, dense_time)
    t0 = dense_time[np.argmax(dense_V)]
    A0 = np.max(dense_V)

    if return_params:
        return A0, t0

    elif return_fit:
        return dense_time, dense_V

    else:
        fit_time = raw_time - t0
        normal_V = raw_real / A0
        return fit_time, normal_V

def setup_r(r, size):

    # Interpret distance domain
    if isinstance(r, numbers.Number):
        r = np.linspace(1, r, size)

    elif isinstance(r, Sized):
        if len(r) == 2:
            r = np.linspace(r[0], r[1], size)
        else:
            r = np.asarray(r)

    else:
        raise ValueError("Unrecognized value for distance domain.")

    return r
