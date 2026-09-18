# Problem 2

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

S0 = 120
K = 115
T = 0.5
r = 0.03
C_market = 8.75


def call_price(sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def objective(sigma):
    return call_price(sigma) - C_market


implied_vol = brentq(objective, 0.0001, 5.0)

print(implied_vol)
print(implied_vol * 100)
