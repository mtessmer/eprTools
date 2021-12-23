import cvxopt as cvo
import numpy as np
from scipy.optimize import nnls
import quadprog

NNLS_FUNCS={}
def nnls_function(func):
    NNLS_FUNCS[func.__name__] = func
    return func

@nnls_function
def cvxnnls(K, L, V, alpha, reltol=1e-8, abstol=1e-9):
    """
    :param KtK:
    :param KtV:
    :param reltol:
    :param abstol:
    :return:
    """

    KtK = K.T @ K + alpha ** 2 * L.T @ L
    KtV = - K.T @ V.T

    # get unconstrained solution as starting point.
    P = np.linalg.inv(KtK) @ KtV
    P = P.clip(min=0)
    P /= P.sum()

    B = cvo.matrix(KtK)

    A = cvo.matrix(KtV)
    C = cvo.matrix(1., (1, len(P)))
    D = cvo.matrix(1.)

    # Minimize with CVXOPT constrained to P >= 0
    lower_bound = cvo.matrix(np.zeros_like(P))
    G = -cvo.matrix(np.eye(len(P), len(P)))
    cvo.solvers.options['show_progress'] = False
    cvo.solvers.options['abstol'] = abstol
    cvo.solvers.options['reltol'] = reltol

    fit_dict = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))
    P = fit_dict['x']
    P = np.squeeze(np.asarray(P))

    return P

@nnls_function
def cvx(K, L, V, alpha, reltol=1e-8, abstol=1e-9):
    """
    :param KtK:
    :param KtV:
    :param reltol:
    :param abstol:
    :return:
    """

    KtK = K.T @ K + alpha ** 2 * L.T @ L
    KtV = - K.T @ V.T

    # get unconstrained solution as starting point.
    P = np.linalg.inv(KtK) @ KtV
    P = P.clip(min=0)

    B = cvo.matrix(KtK)

    A = cvo.matrix(KtV)

    # Minimize with CVXOPT constrained to P >= 0
    lower_bound = cvo.matrix(np.zeros_like(P))
    G = -cvo.matrix(np.eye(len(P), len(P)))
    cvo.solvers.options['show_progress'] = False
    cvo.solvers.options['abstol'] = abstol
    cvo.solvers.options['reltol'] = reltol
    fit_dict = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))
    P = fit_dict['x']
    P = np.squeeze(np.asarray(P))

    return P


@nnls_function
def spnnls(K, L, V, alpha):
    C = np.concatenate([K, alpha * L])
    d = np.concatenate([V, np.zeros(L.shape[0])])
    P, norm = nnls(C, d)
    return P

@nnls_function
def qpnnls(K, L, V, alpha, reltol=1e-8, abstol=1e-9):
    """
    """

    KtK = K.T @ K + alpha ** 2 * L.T @ L
    KtV = K.T @ V.T

    # get unconstrained solution as starting point.
    P = np.linalg.inv(KtK) @ KtV
    P = P.clip(min=0)

    # Minimize with CVXOPT constrained to P >= 0
    lower_bound = np.zeros_like(P)
    G = np.eye(len(P), len(P))


    fit = quadprog.solve_qp(KtK, KtV, G, lower_bound)
    P = fit[0]

    return P