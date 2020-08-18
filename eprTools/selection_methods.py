import numpy as np

SELECTION_METHODS = {}
def selection_method(func):
    SELECTION_METHODS[func.__name__.lower()] = func
    return func

@selection_method
def AIC(K, L, alpha, residuals):
    K_alpha = np.linalg.inv((K.T @ K + alpha ** 2 * L.T @ L)) @ K.T
    H_alpha = K @ K_alpha
    nt = K.shape[1]
    score = nt * np.log((residuals @ residuals) / nt) + (2 * np.trace(H_alpha))

    return score

@selection_method
def GCV(K, L, alpha, residuals):
    K_alpha = np.linalg.inv((K.T @ K + alpha ** 2 * L.T @ L)) @ K.T
    H_alpha = K @ K_alpha
    nt = K.shape[1]
    score = (residuals @ residuals) / ((1 - np.trace(H_alpha) / nt) ** 2)

    return score