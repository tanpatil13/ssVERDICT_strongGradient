import numpy as np
from scipy import special

def computeVerdictSignal(params, b_values, Delta, delta, gradient_strength):
    f_ic = params['f_ic'].value
    f_ees = params['f_ees'].value
    r = params['r'].value
    d_ees = params['d_ees'].value

    SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035]
    
    # d_ees = 2
    d_ic, d_vasc = 2, 8
    alpha = SPHERE_TRASCENDENTAL_ROOTS / r
    alpha2 = alpha ** 2
    alpha2D = alpha2 * d_ic
    alpha = alpha.reshape(1, -1)
    alpha2 = alpha2.reshape(1, -1)
    alpha2D = alpha2D.reshape(1, -1)

    delta = delta.reshape(-1, 1)
    Delta = Delta.reshape(-1, 1)
    b_values = np.array(b_values)


    gamma = 2.675987e2
    first_factor = -2 * (gamma *  gradient_strength) **2 / d_ic

    summands = (alpha ** (-4) / (alpha2 * r**2 - 2) * (
                            2 * delta - (
                            2 +
                            np.exp(-alpha2D * (Delta - delta)) -
                            2 * np.exp(-alpha2D * delta) -
                            2 * np.exp(-alpha2D * Delta) +
                            np.exp(-alpha2D * (Delta + delta))
                        ) / (alpha2D)
                    )
                )

    S_vasc = (1 - f_ic - f_ees) * ((np.sqrt(np.pi) * special.erf(np.sqrt(b_values * d_vasc))) /
                (2 * np.sqrt(b_values * d_vasc)))                                 # astrosticks compartment
    S_ic = f_ic * np.exp(first_factor * np.sum(summands, axis=1))                 # sphere compartment
    S_ees = f_ees * np.exp(-b_values * d_ees)                                     # ball compartment
    S_pred = S_vasc + S_ic + S_ees

    return S_pred

def verdictResiduals(params, S_measured, b_values, Delta, delta, gradient_strength):
    S_pred = computeVerdictSignal(params, b_values, Delta, delta, gradient_strength)
    res = S_measured - S_pred
    return res
