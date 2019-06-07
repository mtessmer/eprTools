import numpy as np
from scipy.special import fresnel


def fredholm_kernel(rmin=15, rmax=80, time=3500, size=200):

    if time < 10:
        time = time * 1000

    r = np.linspace(rmin, rmax, size)
    time = np.linspace(0.01, time, size)

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
    trigterm = np.outer(time, omega_dd)
    z = np.sqrt((6 * trigterm / np.pi))
    S_z, C_z = fresnel(z)
    SzNorm = S_z / z
    CzNorm = C_z / z

    costerm = np.cos(trigterm)
    sinterm = np.sin(trigterm)
    K = CzNorm * costerm + SzNorm * sinterm

    return K

def background(a, d = 3, time = 3500, size = 200):
    if time < 10:
        time = time * 1000

    time = np.linspace(0, time, size)

    return a * np.exp(-k * (time ** (d / 3)))