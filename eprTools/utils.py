import numbers
from collections.abc import Sized
import numpy as np
from numba import njit
from scipy.special import fresnel
from scipy.optimize import curve_fit


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


    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = 256

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

    # Interpret time domain
    if isinstance(time, numbers.Number):
        time = np.linspace(0, time, size)
    elif isinstance(r, Sized):
        if len(time) == 2:
            r = np.linspace(time[0], time[1], size)

        else:
            time = np.asarray(time)

    else:
        raise ValueError("Unrecognized value for time domain.")

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
    trigterm = np.outer(np.abs(time), omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))
    S_z, C_z = fresnel(z) / z
    K = C_z * np.cos(trigterm) + S_z * np.sin(trigterm)

    # Correct for error introduced by divide by zero error
    K[time == 0] = 1

    return K, r, time


def homogeneous_3d(t, k, a, d=3):
    """
    Homogeneous n-dimensional background function to be used for background fitting.
    :param t: numpy ndarray
        time axis

    :param k: float
        fit parameter

    :param a: float
        1 - modulation_depth

    :param d: float
        number of dimensions

    """
    return a * np.exp(-k * (t ** (d / 3)))


def generate_background(a, k, j, d=3, time=3500, size=200):
    if time < 10:
        time = time * 1000

    time = np.linspace(0, time, size)
    return a * np.exp(-k * (time ** (d / 3)) + j)


def read_param_file(param_file):
    param_dict = {}
    with open(param_file, 'r') as file:
        for line in file:
            # Skip blank lines and lines with comment chars
            if line.startswith(("*", "#", "\n")):
                continue

            # Add keywords to param_dict
            line = line.split()
            try:
                key = line.pop(0)
                val = [arg.strip() for arg in line]
            except IndexError:
                key = line
                val = None

            if key:
                param_dict[key] = val

    return param_dict


def fit_nd_background(s, t, fit_start):

    fit_time = t[fit_start:]
    fit_real = s[fit_start:]
    try:
        popt, pcov = curve_fit(homogeneous_3d, fit_time, fit_real,
                               p0=(1e-5, 0.7), bounds=[(1e-7, 0.4), (1e-1, 1)])
    except RuntimeError:
        import matplotlib.pyplot as plt

        plt.plot(fit_time, fit_real)
        plt.show()
        raise RuntimeError

    background = homogeneous_3d(t, *popt)
    mod_depth = 1 - popt[1]

    return background, mod_depth
