import numpy as np
from numba import njit
from scipy.special import fresnel
from scipy.optimize import curve_fit


def generate_kernel(rmin=15, rmax=80, time=3500, size=200):
    """
    Generate DEER kernel using  angstroms and nanoseconds

    :param rmin: float
        minimum distance covered by kernel (Å)

    :param rmax: float
        maximum distance covered by kernel (Å)

    :param time: float
        maximum time covered by kernel (ns)

    :param size: int
        axis size of kernel matrix

    :return: numpy ndarray
        DEER Kernel matrix
    """

    # Ensure units to ns
    if time < 10:
        time = time * 1000

    r = np.linspace(rmin, rmax, size)
    time = np.linspace(0, time, size)

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
    trigterm = np.outer(time, omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))
    S_z, C_z = fresnel(z) / z
    K = C_z * np.cos(trigterm) + S_z * np.sin(trigterm)

    # Correct for error introduced by divide by zero error
    K[0] = 1

    return K


def generate_kernel_nm(rmin=1.5, rmax=8.0, time=3.5, size=200):
    """
    Generate DEER kernel using  nanometers and microsecond

    :param rmin: float
        minimum distance covered by kernel (nm)

    :param rmax: float
        maximum distance covered by kernel (nm)

    :param time: float
        maximum time covered by kernel (μs)

    :param size: int
        axis size of kernel matrix

    :return: numpy ndarray
        DEER Kernel matrix

    """
    r = np.linspace(rmin, rmax, size)
    time = np.linspace(0, time, size)

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
    trigterm = np.outer(time, omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))
    S_z, C_z = fresnel(z) / z
    K = C_z * np.cos(trigterm) + S_z * np.sin(trigterm)

    # Correct for error introduced by divide by zero error
    K[0] = 1

    return K

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
