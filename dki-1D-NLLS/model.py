import numpy as np
from scipy import special

def computeDKISignal(params, b_values):
    D_k = params['D_k'].value
    K = params['K'].value

    b_values = np.array(b_values)

    exponent = (-b_values * D_k) + (b_values**2 * D_k**2 * K)/6
    S_pred = np.exp(exponent)

    return S_pred

def dkiResiduals(params, S_measured, b_values):
    S_pred = computeDKISignal(params, b_values)
    res = S_measured - S_pred
    return res