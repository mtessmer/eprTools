import numpy as np
from scipy.optimize import nnls
import cvxopt as cvx
from sklearn.linear_model import ElasticNetCV


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

    B = cvx.matrix(KtK)

    A = cvx.matrix(KtV)
    C = cvx.matrix(1., (1, len(P)))
    D = cvx.matrix(1.)

    # Minimize with CVXOPT constrained to P >= 0
    lower_bound = cvx.matrix(np.zeros_like(P))
    G = -cvx.matrix(np.eye(len(P), len(P)))
    cvx.solvers.options['show_progress'] = False
    cvx.solvers.options['abstol'] = abstol
    cvx.solvers.options['reltol'] = reltol

    fit_dict = cvx.solvers.qp(B, A, G, lower_bound, C, D, initvals=cvx.matrix(P))
    P = fit_dict['x']
    P = np.squeeze(np.asarray(P))

    return P.clip(0)


@nnls_function
def spnnls(K, L, V, alpha):
    C = np.concatenate([K, alpha * L])
    d = np.concatenate([V, np.zeros(L.shape[0])])
    P, norm = nnls(C, d)
    return P

@nnls_function
def cvxelastic(K, V, alpha, ratio, reltol=1e-8, abstol=1e-9):
    """
    :param KtK:
    :param KtV:
    :param reltol:
    :param abstol:
    :return:
    """
    sz = K.shape[1]
    KtK = K.T @ K  # + (alpha **2) * np.eye(sz)
    KtV = - K.T @ V.T

    # get unconstrained solution as starting point.
    P = np.linalg.inv(KtK) @ KtV
    P = P.clip(min=0)
    P /= P.sum()

    B = cvx.matrix(KtK)

    A = cvx.matrix(KtV)
    C = cvx.matrix(1., (1, len(P)))
    D = cvx.matrix(1.)

    # Minimize with CVXOPT constrained to P >= 0
    lower_bound = cvx.matrix(np.zeros_like(P))
    G = -cvx.matrix(np.eye(len(P), len(P)))
    cvx.solvers.options['show_progress'] = False
    cvx.solvers.options['abstol'] = abstol
    cvx.solvers.options['reltol'] = reltol

    fit_dict = cvx.solvers.qp(B, A, G, lower_bound, C, D, initvals=cvx.matrix(P))
    P = fit_dict['x']
    P = np.squeeze(np.asarray(P))

    return P


@nnls_function
def elastic(K, y):
    eNet = ElasticNetCV(alphas=np.logspace(-4, 4), l1_ratio=np.linspace(0, 1, 11))
    eNet.Vfit(K, y)
    return eNet.coef_, eNet.predict(K), eNet.alpha_, eNet.l1_ratio_
