import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import norm


tickers = ["MSFT", "AAPL"]

prices = pd.DataFrame()

for ticker in tickers:
    data = yf.download(
        ticker,
        start="2011-08-31",
        end="2016-09-02",
        auto_adjust=True,
        progress=False
    )
    prices[ticker] = data["Close"].squeeze()

returns = np.log(prices / prices.shift(1)).dropna()

market_caps = np.array([448.77, 575.11])
weights = market_caps / market_caps.sum()

V = 1_000_000
theta = V * weights

alpha = 0.95
K = 10
lam = 0.97
M = 50_000

print("Portfolio weights:", weights)
print("Observations:", len(returns))


X = returns.to_numpy()

mu = np.zeros(2)
Sigma = np.zeros((2, 2))

def ewma_update(mu, Sigma, x):
    mu_new = lam * mu + (1 - lam) * x
    error = x - mu
    Sigma_new = lam * Sigma + (1 - lam) * np.outer(error, error)
    return mu_new, Sigma_new

for x in X:
    mu, Sigma = ewma_update(mu, Sigma, x)

print("Mean vector:")
print(mu)

print("\nCovariance matrix:")
print(Sigma)


mu_L = -theta @ mu
sigma_L = np.sqrt(theta @ Sigma @ theta)

VaR_1 = mu_L + norm.ppf(alpha) * sigma_L
VaR_10 = np.sqrt(K) * VaR_1

RC = 3 * VaR_10

print(f"1-day VaR: ${VaR_1:,.2f}")
print(f"10-day VaR: ${VaR_10:,.2f}")
print(f"Regulatory capital: ${RC:,.2f}")


sigma = np.sqrt(np.diag(Sigma))
rho = Sigma[0, 1] / (sigma[0] * sigma[1])

# Apple shock
x2 = mu[1] - 5 * sigma[1]

# Microsoft conditional distribution
mu_cond = mu[0] + rho * sigma[0] / sigma[1] * (x2 - mu[1])
sigma_cond = sigma[0] * np.sqrt(1 - rho**2)

# Microsoft shock
x1 = mu_cond - 5 * sigma_cond

shock = np.array([x1, x2])

# Updated EWMA parameters
mu_shock, Sigma_shock = ewma_update(mu, Sigma, shock)

print("Shocked returns:", shock)
print("Updated mean:", mu_shock)
print("Updated covariance:")
print(Sigma_shock)


np.random.seed(42)

losses = np.zeros(M)

for m in range(M):

    mu_sim = mu_shock.copy()
    Sigma_sim = Sigma_shock.copy()

    total_return = shock.copy()

    for k in range(2, K + 1):

        x = np.random.multivariate_normal(mu_sim, Sigma_sim)
        total_return += x

        if k < K:
            mu_sim, Sigma_sim = ewma_update(mu_sim, Sigma_sim, x)

    losses[m] = -theta @ total_return

print("Simulations completed:", M)


avg_loss = np.mean(losses)
VaR_shocked = np.quantile(losses, alpha)

exceed_VaR = 100 * np.mean(losses > VaR_10)
exceed_RC = 100 * np.mean(losses > RC)

results = pd.DataFrame({
    "Metric": [
        "(a) Average loss",
        "(b) Shocked 10-day VaR",
        "(c) Initial VaR exceedance",
        "(d) Regulatory capital exceedance"
    ],
    "Value": [
        f"${avg_loss:,.2f}",
        f"${VaR_shocked:,.2f}",
        f"{exceed_VaR:.2f}%",
        f"{exceed_RC:.2f}%"
    ]
})

print(results.to_string(index=False))


plt.figure(figsize=(9, 5))

plt.hist(losses, bins=70, density=True, alpha=0.7)

plt.axvline(VaR_10, color="blue", linestyle="--", label="Initial VaR")
plt.axvline(VaR_shocked, color="red", linestyle="--", label="Shocked VaR")
plt.axvline(RC, color="green", linestyle="--", label="Regulatory Capital")

plt.xlabel("10-day Portfolio Loss ($)")
plt.ylabel("Density")
plt.title("Stressed Portfolio Loss Distribution")
plt.legend()
plt.tight_layout()

plt.savefig("stress_testing.png", dpi=300, bbox_inches="tight")
plt.show()
