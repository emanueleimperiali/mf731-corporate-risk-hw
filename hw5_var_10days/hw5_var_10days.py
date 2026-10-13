import numpy as np
import pandas as pd
from scipy.stats import norm
from IPython.display import display

S0 = 152.51
K = 170
T = 0.292

r = 0.00119
sigma = 0.4907
mu = 0.1691

dt = 1 / 252
days = 10

M = 100
N = 40_000
alpha = 0.95


def put_price(S, tau):

    d1 = (
        np.log(S / K) +
        (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))

    d2 = d1 - sigma * np.sqrt(tau)

    return (
        K * np.exp(-r * tau) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


def put_delta(S, tau):

    d1 = (
        np.log(S / K) +
        (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))

    return norm.cdf(d1) - 1


P0 = put_price(S0, T)

h0 = put_delta(S0, T)

cash0 = -h0 * S0

V0 = h0 * S0 + cash0 - P0

print("Initial put price:", round(P0, 4))
print("Initial put delta:", round(h0, 4))
print("Initial portfolio value:", round(M * V0, 2))


np.random.seed(42)

# Simulate one-day stock prices
Z = np.random.normal(0, 1, N)

S1 = S0 * np.exp(
    (mu - 0.5 * sigma**2) * dt
    + sigma * np.sqrt(dt) * Z
)

# Reprice the put
P1 = put_price(S1, T - dt)

# Update cash account
cash1 = cash0 * np.exp(r * dt)

# Final portfolio value
V1 = h0 * S1 + cash1 - P1

# Loss distribution
loss_1day = M * (V0 - V1)

# VaR calculations
var_1day = np.quantile(loss_1day, alpha)

var_10day_scaled = np.sqrt(days) * var_1day

print("One-day VaR: $", round(var_1day, 2))
print("Scaled ten-day VaR: $", round(var_10day_scaled, 2))


np.random.seed(43)

# Initialize portfolios
S = np.full(N, S0)

h = np.full(N, h0)

cash = np.full(N, cash0)

# Simulate ten trading days
for day in range(1, days + 1):

    Z = np.random.normal(0, 1, N)

    # Simulate stock price
    S = S * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    # Grow cash account
    cash = cash * np.exp(r * dt)

    # Reprice put
    tau = T - day * dt

    P = put_price(S, tau)

    # Rebalance delta hedge
    if day < days:

        h_new = put_delta(S, tau)

        stock_change = h_new - h

        cash = cash - stock_change * S

        h = h_new

# Final portfolio value
V10 = h * S + cash - P

# Ten-day losses
loss_10day = M * (V0 - V10)

# Ten-day VaR
var_10day_simulated = np.quantile(loss_10day, alpha)

print("Simulated ten-day VaR: $", round(var_10day_simulated, 2))


results = pd.DataFrame({
    "Method": [
        "One-day Monte Carlo",
        "Ten-day sqrt scaling",
        "Ten-day direct simulation"
    ],
    "95% VaR ($)": [
        var_1day,
        var_10day_scaled,
        var_10day_simulated
    ]
})

display(results.round(2))

difference = var_10day_scaled - var_10day_simulated

percentage = difference / var_10day_simulated * 100

print("Absolute difference: $", round(difference, 2))
print("Percentage difference:", round(percentage, 2), "%")
