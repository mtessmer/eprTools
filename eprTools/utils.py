import numpy as np
from scipy.special import fresnel


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
    time = np.linspace(0.01, time, size)

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
    trigterm = np.outer(time, omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))

    # Set zero elements to 1 to avoid divide by zero error
    z[z == 0] = 1
    S_z, C_z = fresnel(z)
    SzNorm = S_z / z
    CzNorm = C_z / z

    costerm = np.cos(trigterm)
    sinterm = np.sin(trigterm)
    K = CzNorm * costerm + SzNorm * sinterm

    # Correct for error introduced by avoiding divide by zero error
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

    # Set zero elements to 1 to avoid divide by zero error
    z[z == 0] = 1
    S_z, C_z = fresnel(z)
    SzNorm = S_z / z
    CzNorm = C_z / z
    costerm = np.cos(trigterm)
    sinterm = np.sin(trigterm)
    K = CzNorm * costerm + SzNorm * sinterm

    # Correct for error introduced by avoiding divide by zero error
    K[0] = 1

    return K

def homogeneous_3d(t, k, a, d):
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
            else:
                line = line.split()
                try:
                    key = line[0]
                    val = [arg.strip() for arg in line[1:]]
                except IndexError:
                    key = line
                    val = None

                if key:
                    param_dict[key] = val

    return param_dict