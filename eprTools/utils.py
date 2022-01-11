import numbers
import warnings
from collections.abc import Sized
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel
from scipy.optimize import minimize
from scipy.linalg import qr


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

    omega_dd = (2 * np.pi * 52.0410) / (r ** 3)
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


def reg_range(K, L, noiselvl=0., logres=0.1):

    # Set alpha range
    minmax_ratio = 16 * np.finfo(float).eps * 1e6  # ratio of smallest to largest alpha
    # Scaling by noise. This improves L curve corner detection for DEER.
    minmax_ratio = minmax_ratio * 2 ** (noiselvl / 0.0025)

    # Get generalized singular values of K and L

    singularValues = gsvd(K, L)

    DerivativeOrder = L.shape[1] - L.shape[0]  # get order of derivative (=number of inf in singval)
    singularValues = singularValues[0:len(singularValues) - DerivativeOrder]  # remove inf
    singularValues = singularValues[::-1]  # sort in decreasing order
    singularValues = singularValues[singularValues > 0]  # remove zeros
    lgsingularValues = np.log10(singularValues)

    # Calculate range based on singular values
    lgrangeMax = lgsingularValues[0]
    lgrangeMin = np.maximum(lgsingularValues[-1], lgsingularValues[0] + np.log10(minmax_ratio))
    lgrangeMax = np.floor(lgrangeMax / logres) * logres
    lgrangeMin = np.ceil(lgrangeMin / logres) * logres
    if lgrangeMax < lgrangeMin:
        temp = lgrangeMax
        lgrangeMax = lgrangeMin
        lgrangeMin = temp
    lgalpha = np.arange(lgrangeMin, lgrangeMax, logres)
    lgalpha = np.append(lgalpha, lgrangeMax)
    alphas = 10 ** lgalpha

    return alphas


def reg_operator(r, kind='L2'):
    loffset, uoffset = (0, None) if kind[-1] == '+' else (1, -1)

    L = np.zeros((len(r), len(r)))
    diag = np.arange(len(r))

    L[diag[:-1], diag[1:]] = 1
    L[diag, diag] = - 2
    L[diag[1:], diag[:-1]] = 1

    return L[loffset:uoffset]


def gsvd(A, B):
    m, p = A.shape
    n = B.shape[0]

    # Economy-sized.
    useQA = m > p
    useQB = n > p
    if useQA:
        QA, A = qr(A)
        A = A[0:p,0:p]
        QA = QA[:,0:p]
        m = p

    if useQB:
        QB, B = qr(B)
        B = B[0:p,0:p]
        QB = QB[:,0:p]
        n = p

    Q, _ = np.linalg.qr(np.vstack((A, B)), mode='reduced')
    Q1 = Q[0:m, 0:p]
    Q2 = Q[m:m+n, 0:p]

    U, _, _, C, S = csd(Q1, Q2)

    # Vector of generalized singular values.
    q = min(m+n, p)

    # Supress divide by 0 warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        U = np.vstack((np.zeros((q-m, 1), 'double'), np.diag(C, max(0, q-m)).reshape(len(np.diag(C, max(0, q-m))), 1))) / np.vstack((np.diag(S, 0).reshape(len(np.diag(S, 0)), 1), np.zeros((q-n,1),'double') ))

    return U


def csd(Q1, Q2):
    """
    Cosine-Sine Decomposition
    U,V,Z,C,S = csd(Q1,Q2)
    Given Q1 and Q2 such that Q1'@Q1 + Q2'@Q2 = I, the
    C-S Decomposition is a joint factorization of the form
    Q1 = U@C@Z' and Q2=V@S@Z'
    where U, V, and Z are orthogonal matrices and C and S
    are diagonal matrices (not necessarily square) satisfying
    C'@C + S'@S = I
    """
    [m, p] = np.shape(Q1)
    n = np.shape(Q2)[0]

    if m < n:
        V, U, Z, S, C = csd(Q2, Q1)
        j = np.flip(np.arange(p))
        C = C[:, j]
        S = S[:, j]
        Z = Z[:, j]
        m = min(m, p)
        i = np.flip(np.arange(m))
        C[np.arange(m), :] = C[i, :]
        U[:, np.arange(m)] = U[:, i]
        n = min(n, p)
        i = np.flip(np.arange(n))
        S[np.arange(n), :] = S[i, :]
        V[:, np.arange(n)] = V[:, i]
        return U, V, Z, C, S

    U, sdiag, VH = np.linalg.svd(Q1)
    C = np.zeros((m, p))
    np.fill_diagonal(C, sdiag)
    Z = VH.T.conj()

    q = min(m, p)
    i = np.arange(0, q, 1)
    j = np.arange(q - 1, -1, -1)
    C[i, i] = C[j, j]
    U[:, i] = U[:, j]
    Z[:, i] = Z[:, j]
    S = Q2 @ Z

    if q == 1:
        k = 0
    elif m < p:
        k = n
    else:
        k = max(0, np.max(np.where(np.diag(C) <= 1 / np.sqrt(2))))
    V, _ = qr(S[:, 0:k])

    S = V.T @ S
    r = min(k, m)
    S[:r - 1, :r - 1] = np.diag(np.diag(S[:, 0:r - 1]))
    S[r:, :r - 1] = 0

    if (m == 1 and p > 1):
        S[0, 0] = 0

    if k < min(n, p):
        r = min(n, p)
        i = np.arange(k, n, 1)
        j = np.arange(k, r, 1)
        [UT, STdiag, VH] = np.linalg.svd(S[np.ix_(i, j)])
        ST = np.zeros((len(i), len(j)))
        np.fill_diagonal(ST, STdiag)
        VT = VH.T.conj()
        if k > 0:
            S[0:k, j] = 0
        S[np.ix_(i, j)] = ST
        C[:, j] = C[:, j] @ VT
        V[:, i] = V[:, i] @ UT
        Z[:, j] = Z[:, j] @ VT
        i = np.arange(k, q, 1)
        [Q, R] = qr(C[np.ix_(i, j)])
        C[np.ix_(i, j)] = np.triu(np.tril(R))

        U[:, i] = U[:, i] @ Q

    if m < p:
        # Diagonalize final block of S and permute blocks.
        eps = np.finfo(float).eps
        q = min([np.count_nonzero(abs(np.diag(C, 0)) > 10 * m * eps),
                 np.count_nonzero(abs(np.diag(S, 0)) > 10 * n * eps),
                 np.count_nonzero(np.amax(abs(S[:, m + 1:p]), axis=1) < np.sqrt(eps))])

        # maxq: maximum size of q such that the expression used later on,
        #        i = [q+1:q+p-m, 1:q, q+p-m+1:n],
        # is still a valid permutation.
        maxq = m + n - p
        q = q + np.count_nonzero(np.amax(abs(S[:, q + 1:maxq + 1]), axis=1) > np.sqrt(eps))

        i = np.arange(q, n, 1)
        j = np.arange(m, p, 1)
        # At this point, S(i,j) should have orthogonal columns and the
        # elements of S(:,q+1:p) outside of S(i,j) should be negligible.
        Q, R = qr(S[np.ix_(i, j)])
        S[:, q + 1:p] = 0
        S[np.ix_(i, j)] = np.diag(np.diag(R))
        V[:, i] = V[:, i] @ Q
        if n > 1:
            i = np.concatenate((np.arange(q, q + p - m, 1), np.arange(0, q, 1), np.arange(q + p - m, n, 1)))
        else:
            i = 1
        j = np.concatenate((np.arange(m, p, 1), np.arange(0, m, 1)))
        C = C[:, j]
        S = S[np.ix_(i, j)]
        Z = Z[:, j]
        V = V[:, i]

    if n < p:
        # Final block of S is negligible.
        S[:, n + 1:p] = 0

    # Make sure C and S are real and positive.
    U, C = diagp(U, C, max(0, p - m))
    C = C.real

    V, S = diagp(V, S, 0)
    S = S.real

    return U, V, Z, C, S

def diagf(X):
    """
    Diagonal force
    X = diagf(X) zeros all the elements off the main diagonal of X.
    """
    X = np.triu(np.tril(X))
    return X


def diagp(Y,X,k):
    """
    DIAGP  Diagonal positive.
    Y,X = diagp(Y,X,k) scales the columns of Y and the rows of X by
    unimodular factors to make the k-th diagonal of X real and positive.
    """
    D = np.diag(X,k)
    j = np.where((D.real < 0) | (D.imag != 0))
    D = np.diag(np.conj(D[j])/abs(D[j]))
    Y[:,j] = Y[:,j]@D.T
    X[j,:] = D@X[j,:]
    X = X+0 # use "+0" to set possible -0 elements to 0
    return Y, X


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

def fit_zero_time(raw_time, raw_real, return_params=False):
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
