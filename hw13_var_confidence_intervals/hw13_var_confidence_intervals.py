import numpy as np
import yfinance as yf
from scipy.stats import norm, chi2

# Parameters
V = 1_000_000
alpha = 0.97
beta = 0.02
M = 125_000

# Historical AAPL data (end date is exclusive)
data = yf.download("AAPL", start="2016-11-01", end="2018-11-01",
                   auto_adjust=False, progress=False)
prices = data["Adj Close"].squeeze()
returns = np.log(prices / prices.shift(1)).dropna().to_numpy()

n = len(returns)
xbar = returns.mean()
s2 = returns.var(ddof=1)
s = np.sqrt(s2)
z = norm.ppf(1 - alpha)

print(f"Observations: {n}")
print(f"Mean: {xbar:.8f}")
print(f"Variance: {s2:.8f}")

# Empirical and theoretical VaR
losses = V * (1 - np.exp(returns))
k = int(np.ceil(n * alpha))
VaR_emp = np.sort(losses)[k - 1]
VaR_theo = V * (1 - np.exp(xbar + s * z))

# Exact 98% CI for variance under normal, iid log returns
df = n - 1
sigma2_low = df * s2 / chi2.ppf(1 - beta / 2, df)
sigma2_high = df * s2 / chi2.ppf(beta / 2, df)

# VaR increases with sigma because z = Phi^{-1}(1-alpha) < 0
VaR_low_a = V * (1 - np.exp(xbar + np.sqrt(sigma2_low) * z))
VaR_high_a = V * (1 - np.exp(xbar + np.sqrt(sigma2_high) * z))

print(f"Empirical VaR: ${VaR_emp:,.2f}")
print(f"Theoretical VaR: ${VaR_theo:,.2f}")
print(f"Part (a) 98% CI: (${VaR_low_a:,.2f}, ${VaR_high_a:,.2f})")

rng = np.random.default_rng(1)

Q = rng.chisquare(df, size=M)
sigma_sim = np.sqrt(df * s2 / Q)
mu_sim = xbar + sigma_sim * rng.standard_normal(M) / np.sqrt(n)

Y = V * (1 - np.exp(mu_sim + sigma_sim * z))
VaR_sim = Y.mean()
VaR_low_b, VaR_high_b = np.quantile(Y, [beta / 2, 1 - beta / 2])

print(f"Simulated VaR: ${VaR_sim:,.2f}")
print(f"Part (b) 98% CI: (${VaR_low_b:,.2f}, ${VaR_high_b:,.2f})")

print(f"Empirical VaR:      ${VaR_emp:,.2f}")
print(f"Theoretical VaR:    ${VaR_theo:,.2f}")
print(f"Part (a) 98% CI:    (${VaR_low_a:,.2f}, ${VaR_high_a:,.2f})")
print(f"Simulated VaR:      ${VaR_sim:,.2f}")
print(f"Part (b) 98% CI:    (${VaR_low_b:,.2f}, ${VaR_high_b:,.2f})")
print(f"CI width (a):       ${VaR_high_a - VaR_low_a:,.2f}")
print(f"CI width (b):       ${VaR_high_b - VaR_low_b:,.2f}")
