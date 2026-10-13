# Emanuele Imperiali

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import norm
from IPython.display import display

tickers = ["MSFT", "AAPL", "GOOGL"]
V = 1_000_000
alpha = 0.95


# 1
# Download daily closing prices for MSFT, AAPL, and GOOGL
# from May 23, 2024 to May 22, 2026.

prices = yf.download(
    tickers,
    start="2024-05-23",
    end="2026-05-23",
    auto_adjust=False,
    progress=False
)["Close"]

prices = prices[tickers].dropna()

print("First five observations:")
display(prices.head())

print("Last five observations:")
display(prices.tail())

print("Total observations:", len(prices))


# 2
# Portfolio weighted by market capitalization as of May 22, 2026.

market_caps = np.array([3.109, 4.536, 4.640])

weights = market_caps / market_caps.sum()

allocations = V * weights

portfolio = pd.DataFrame({
    "Market Cap (Trillion USD)": market_caps,
    "Weight": weights,
    "Allocation (USD)": allocations
}, index=tickers)

display(portfolio)


# 3
# 95% VaR from the empirical distribution of daily log returns.

X = np.log(prices / prices.shift(1)).dropna()

L_full = -V * ((np.exp(X) - 1) @ weights)
L_lin = -V * (X @ weights)
L_quad = -V * ((X + 0.5 * X**2) @ weights)

var_emp = pd.Series({
    "Full": np.quantile(L_full, alpha),
    "Linear": np.quantile(L_lin, alpha),
    "Quadratic": np.quantile(L_quad, alpha)
})

display(var_emp.to_frame("Empirical VaR (USD)"))

plt.figure(figsize=(9, 5))

plt.hist(L_full, bins=40, alpha=0.4, label="Full")
plt.hist(L_lin, bins=40, alpha=0.4, label="Linear")
plt.hist(L_quad, bins=40, alpha=0.4, label="Quadratic")

plt.xlabel("Portfolio Loss (USD)")
plt.ylabel("Frequency")
plt.title("Empirical Loss Distributions")
plt.legend()
plt.show()


# 4
# EWMA estimates of the conditional mean vector and covariance matrix.

lam = 0.97
theta = 0.97

mu = np.zeros(3)
Sigma = np.zeros((3, 3))

for x in X.to_numpy():
    error = x - mu

    Sigma = theta * Sigma + (1 - theta) * np.outer(error, error)
    mu = lam * mu + (1 - lam) * x

print("EWMA Mean:")
display(pd.Series(mu, index=tickers))

print("EWMA Covariance:")
display(pd.DataFrame(Sigma, index=tickers, columns=tickers))


# 5
# 95% VaR assuming multivariate normal log returns
# with the EWMA mean and covariance.

N = 100_000
np.random.seed(42)

X_sim = np.random.multivariate_normal(mu, Sigma, size=N)

L_full_mc = -V * ((np.exp(X_sim) - 1) @ weights)

L_quad_mc = -V * ((X_sim + 0.5 * X_sim**2) @ weights)

var_full_mc = np.quantile(L_full_mc, alpha)
var_quad_mc = np.quantile(L_quad_mc, alpha)

print("Full MC VaR:", var_full_mc)
print("Quadratic MC VaR:", var_quad_mc)

mu_p = weights @ mu
sigma_p = np.sqrt(weights @ Sigma @ weights)

z = norm.ppf(alpha)

var_lin_analytical = -V * mu_p + z * V * sigma_p

print("Analytical Linear VaR:", var_lin_analytical)

var_param = pd.Series({
    "Full": var_full_mc,
    "Linear": var_lin_analytical,
    "Quadratic": var_quad_mc
})

display(var_param.to_frame("Parametric VaR (USD)"))


# 6
# Comparison of empirical and parametric VaR estimates.

comparison = pd.DataFrame({
    "Empirical VaR": var_emp,
    "Parametric VaR": var_param
})

comparison["Difference (%)"] = (
    (comparison["Parametric VaR"] / comparison["Empirical VaR"] - 1) * 100
)

display(comparison.round(2))

comparison[["Empirical VaR", "Parametric VaR"]].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("95% VaR Comparison")
plt.ylabel("VaR (USD)")
plt.xlabel("Loss Operator")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
