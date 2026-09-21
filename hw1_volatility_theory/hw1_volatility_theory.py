# ============================================================
# Emanuele Imperiali - HW 1
# ============================================================


# ============================================================
# 1
# ============================================================

import numpy as np

sigma_daily = 0.02

sigma_3days = sigma_daily * np.sqrt(3)

print(f"3-day volatility: {sigma_3days:.4%}")


# ============================================================
# 2
# ============================================================

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

print(f"Implied volatility: {implied_vol:.4%}")


# ============================================================
# 3
# ============================================================

sigma_old = 0.015
S_old = 30.00
S_new = 30.50
lam = 0.94

X = np.log(S_new / S_old)

variance_new = lam * sigma_old**2 + (1 - lam) * X**2
sigma_new = np.sqrt(variance_new)

print(f"Daily return: {X:.4%}")
print(f"Updated volatility: {sigma_new:.4%}")


# ============================================================
# 4
# ============================================================

S_old = 300
S_new = 298
sigma_old = 0.013

X = np.log(S_new / S_old)

# (a) EWMA
lam = 0.94

variance_ewma = lam * sigma_old**2 + (1 - lam) * X**2
sigma_ewma = np.sqrt(variance_ewma)

# (b) GARCH(1,1)
alpha0 = 0.000002
alpha1 = 0.04
beta1 = 0.94

variance_garch = alpha0 + alpha1 * X**2 + beta1 * sigma_old**2
sigma_garch = np.sqrt(variance_garch)

print(f"Daily return: {X:.4%}")
print(f"EWMA volatility: {sigma_ewma:.4%}")
print(f"GARCH volatility: {sigma_garch:.4%}")


# ============================================================
# 5
# ============================================================

alpha0 = 0.0000013465
alpha1 = 0.083394
beta1 = 0.910116

VL = alpha0 / (1 - alpha1 - beta1)
sigma_L = np.sqrt(VL)

print(f"Long-term variance: {VL:.8f}")
print(f"Long-term volatility: {sigma_L:.4%}")


# ============================================================
# (b)
# ============================================================

sigma_today = 0.01732
a = alpha1 + beta1

def expected_variance(n):
    return VL + a**n * (sigma_today**2 - VL)

variance_10 = expected_variance(10)
variance_500 = expected_variance(500)

sigma_10 = np.sqrt(variance_10)
sigma_500 = np.sqrt(variance_500)

print(f"Expected volatility after 10 days: {sigma_10:.4%}")
print(f"Expected volatility after 500 days: {sigma_500:.4%}")


# ============================================================
# (c)
# ============================================================
