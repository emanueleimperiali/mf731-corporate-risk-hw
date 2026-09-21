# ============================================================
# HW 1 - Volatility of SPY (MA, EWMA, GARCH(1,1))
# ============================================================


# ============================================================
# Task 1 - MA(100) and EWMA(0.94) annualized volatility
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# 1. parameters
n = 100
lam = 0.94
delta = 1 / 252

# 2. download prices (end is exclusive -> 2026-07-31 includes 2026-07-30)
prices = yf.download("SPY", start="2021-08-01", end="2026-07-31", progress=False)["Close"].squeeze()

# 3. log returns
x = np.log(prices / prices.shift(1)).dropna()

# 4. MA(n) daily volatility
ma_vol = x.rolling(n).std()

# 5. EWMA daily volatility (initial variance = sample variance of the first n returns)
ewma_var = pd.Series(np.nan, index=x.index)
ewma_var.iloc[n - 1] = x.iloc[:n].var()
for i in range(n, len(x)):
    ewma_var.iloc[i] = lam * ewma_var.iloc[i - 1] + (1 - lam) * x.iloc[i - 1] ** 2
ewma_vol = np.sqrt(ewma_var)

# 6. annualize
ma_ann = ma_vol / np.sqrt(delta)
ewma_ann = ewma_vol / np.sqrt(delta)

# 7. plot
plt.figure(figsize=(12, 6))
plt.plot(ma_ann, label=f"MA (n={n})")
plt.plot(ewma_ann, label=f"EWMA (lambda={lam})")
plt.xlabel("Date")
plt.ylabel("Annualized volatility")
plt.title("SPY annualized volatility: MA vs EWMA")
plt.legend()
plt.show()


# ============================================================
# Task 2 - Fit GARCH(1,1) by MLE (06/01/2025 - 05/31/2026)
# ============================================================

# 1. returns in the fitting window (in %)
xf = x.loc["2025-06-01":"2026-05-31"].values * 100
N = len(xf)

# 2. negative log-likelihood
def garch_variances(theta, xf):
    a0, a1, b1 = theta
    var = np.empty(len(xf))
    var[0] = xf.var(ddof=1)
    for i in range(1, len(xf)):
        var[i] = a0 + a1 * xf[i - 1] ** 2 + b1 * var[i - 1]
    return var

def neg_loglik(theta, xf):
    var = garch_variances(theta, xf)
    return np.sum(np.log(var) + xf ** 2 / var)

# 3. minimize
res = minimize(
    neg_loglik, x0=[0.05, 0.05, 0.90], args=(xf,),
    bounds=[(1e-8, None), (0, 1), (0, 1)],
    constraints={"type": "ineq", "fun": lambda t: 0.9999 - t[1] - t[2]},
    method="SLSQP",
)
a0, a1, b1 = res.x
print(f"alpha0 = {a0 / 1e4:.4e}  (= {a0:.4f} in %^2)")
print(f"alpha1 = {a1:.4f}")
print(f"beta1  = {b1:.4f}")
print(f"long-run annualized vol = {np.sqrt(a0 / (1 - a1 - b1) / 1e4 / delta):.4f}")


# ============================================================
# Task 3 - Forecast with $M = 100$ simulated paths (06/01/2026 - 07/30/2026)
# ============================================================

# 1. setup
M = 100
dates = x.loc["2026-06-01":"2026-07-30"].index
m = len(dates)
rng = np.random.default_rng(0)

# 2. starting point
var_N = garch_variances(res.x, xf)[-1]
x_N = xf[-1]

# 3. simulate M paths
paths = np.empty((M, m))
for k in range(M):
    var_prev, x_prev = var_N, x_N
    for j in range(m):
        var_j = a0 + a1 * x_prev ** 2 + b1 * var_prev
        x_j = np.sqrt(var_j) * rng.standard_normal()
        paths[k, j] = np.sqrt(var_j)
        var_prev, x_prev = var_j, x_j

# 4. back to decimals and annualize
paths_ann = paths / 100 / np.sqrt(delta)

plt.figure(figsize=(12, 6))
for k in range(M):
    plt.plot(dates, paths_ann[k], color="gray", alpha=0.15)
plt.plot([], [], color="gray", label="GARCH(1,1) simulated paths")
plt.plot(dates, ma_ann.loc[dates], color="blue", lw=2, label=f"MA (n={n})")
plt.plot(dates, ewma_ann.loc[dates], color="red", lw=2, label=f"EWMA (lambda={lam})")
plt.xlabel("Date")
plt.ylabel("Annualized volatility")
plt.title("GARCH(1,1) forecast paths vs MA and EWMA (06/01/2026 - 07/30/2026)")
plt.legend()
plt.show()


# ============================================================
# Comparison and comments
# ============================================================
