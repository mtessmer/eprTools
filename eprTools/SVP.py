import numpy as np
from scipy.optimize._numdiff import approx_derivative


def SVP(func, x0, lb=(), ub=(), ftol=1e-8, xtol=1e-8, maxiter=100):
    """Solve snlls problem using the Secant Projection Method (Song et al. 2020)
    :param func:
    :param x0:
    :param lb:
    :param ub:
    :param ftol:
    :param xtol:
    :param maxiter:

    :returns

    """
    x0 = np.asarray(x0)
    x = x0.copy()
    lb, ub = np.asarray(lb), np.asarray(ub)
    Jac = approx_derivative(func, x)
    res = func(x)

    for i in range(maxiter):

        # Calcuate gradient step
        d = -np.linalg.inv(Jac.T @ Jac + np.eye(len(x))) @ (Jac.T @ res)

        # Apply step, and ensure it is withing the bounds
        x += d
        x = np.maximum(x, lb)
        x = np.minimum(x, ub)

        # Calculate new residuals
        resn = func(x)

        # Apply wolfe condition
        cl = 1e-4
        w_iter = 0
        while (resn @ resn) > (res @ res + 2 * cl * res @ Jac @ d) and w_iter < 20:
            d *= 0.1
            w_iter += 1

        # Check ftol
        if np.all(np.abs(res - resn) < ftol):
            print("FTOL")
            break

        # Update Jacobian
        uk = resn - res
        num = np.outer((uk - Jac @ d), d)
        Jac += num / (d @ d)
        res = resn

        # Check xtol
        if np.linalg.norm(d) < xtol * (xtol + np.linalg.norm(x)):
            print('XTOL')
            break
    print(f"Converged in {i} iterations")
    print(d)
    return