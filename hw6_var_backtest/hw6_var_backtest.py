# Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math

from scipy.stats import norm, binom
from IPython.display import display

# Parameters

alpha = 0.95
beta = 0.05
M = 1010
lam = 0.97
theta = 0.97

# Plot settings

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True


ticker = "^GSPC"

data = yf.download(
    ticker,
    start="2016-05-20",
    end="2026-05-23",
    auto_adjust=False,
    progress=False
)

close_all = data["Close"].squeeze()

# Dataset required in the homework
prices = close_all.loc["2016-05-23":"2026-05-22"].to_frame("Close")

N = len(prices)

print("First five observations:")
display(prices.head())

print("Last five observations:")
display(prices.tail())

print("N =", N)


# One log return for each date in the required dataset
log_returns = np.log(close_all / close_all.shift(1))
log_returns = log_returns.loc[prices.index]
log_returns.name = "Log Return"

print("Number of log returns =", len(log_returns))


alpha = 0.95
beta = 0.05
M = 1010

losses = 1 - np.exp(log_returns)


def empirical_var(sample, alpha):
    ordered = np.sort(sample)
    k = math.ceil(alpha * len(ordered))
    return ordered[k - 1]


var_emp = []
loss_emp = []
exceed_emp = []
dates = []

for i in range(M, N):
    
    # Previous M losses only
    window = losses.iloc[i-M:i]
    
    # VaR estimated at the beginning of the evaluation period
    var_i = empirical_var(window, alpha)
    
    # Realized loss
    loss_i = losses.iloc[i]
    
    # Exceedance
    exceed_i = loss_i > var_i
    
    dates.append(losses.index[i])
    var_emp.append(var_i)
    loss_emp.append(loss_i)
    exceed_emp.append(exceed_i)


results_emp = pd.DataFrame({
    "Loss": loss_emp,
    "VaR": var_emp,
    "Exceedance": exceed_emp
}, index=dates)

results_emp.index.name = "Date"

J = len(results_emp)

print("J =", J)
print("N - M =", N - M)

display(results_emp.head())


plt.figure(figsize=(12, 5))

plt.plot(
    results_emp.index,
    results_emp["Loss"],
    label="Realized Loss",
    linewidth=0.8
)

plt.plot(
    results_emp.index,
    results_emp["VaR"],
    label="Empirical VaR",
    linewidth=1.2
)

exceedances = results_emp[results_emp["Exceedance"]]

plt.scatter(
    exceedances.index,
    exceedances["Loss"],
    marker="x",
    label="Exceedance"
)

plt.xlabel("Date")
plt.ylabel("Loss")
plt.title("Empirical VaR Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


K_emp = results_emp["Exceedance"].sum()

print("Empirical exceedances =", K_emp)
print("Exceedance rate =", K_emp / J)


lam = 0.97
theta = 0.97

# Initial estimates from first M log returns
mu = log_returns.iloc[:M].mean()
variance = log_returns.iloc[:M].var(ddof=1)

print("Initial mu =", mu)
print("Initial variance =", variance)
print("Initial sigma =", np.sqrt(variance))


var_ewma = []
loss_ewma = []
exceed_ewma = []
mu_series = []
sigma_series = []
dates = []

z = norm.ppf(1 - alpha)

for i in range(M, N):
    
    sigma = np.sqrt(variance)
    
    # VaR forecast before observing return i
    var_i = 1 - np.exp(mu + sigma * z)
    
    # Realized return and loss
    x_i = log_returns.iloc[i]
    loss_i = 1 - np.exp(x_i)
    
    # Exceedance
    exceed_i = loss_i > var_i
    
    dates.append(log_returns.index[i])
    var_ewma.append(var_i)
    loss_ewma.append(loss_i)
    exceed_ewma.append(exceed_i)
    mu_series.append(mu)
    sigma_series.append(sigma)
    
    # EWMA update
    mu_old = mu
    
    variance = (
        theta * variance
        + (1 - theta) * (x_i - mu_old)**2
    )
    
    mu = (
        lam * mu_old
        + (1 - lam) * x_i
    )


results_ewma = pd.DataFrame({
    "Loss": loss_ewma,
    "VaR": var_ewma,
    "Mu": mu_series,
    "Sigma": sigma_series,
    "Exceedance": exceed_ewma
}, index=dates)

results_ewma.index.name = "Date"

display(results_ewma.head())


assert len(results_ewma) == J
assert results_ewma.index.equals(results_emp.index)


K_ewma = results_ewma["Exceedance"].sum()

print("EWMA exceedances =", K_ewma)
print("Exceedance rate =", K_ewma / J)


plt.figure(figsize=(12, 5))

plt.plot(
    results_ewma.index,
    results_ewma["Loss"],
    label="Realized Loss",
    linewidth=0.8
)

plt.plot(
    results_ewma.index,
    results_ewma["VaR"],
    label="Normal-EWMA VaR",
    linewidth=1.2
)

exceedances_ewma = results_ewma[results_ewma["Exceedance"]]

plt.scatter(
    exceedances_ewma.index,
    exceedances_ewma["Loss"],
    marker="x",
    label="Exceedance"
)

plt.xlabel("Date")
plt.ylabel("Loss")
plt.title("Normal-EWMA VaR Backtest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


plt.figure(figsize=(12, 5))

plt.plot(
    results_emp.index,
    results_emp["VaR"],
    label="Empirical VaR"
)

plt.plot(
    results_ewma.index,
    results_ewma["VaR"],
    label="Normal-EWMA VaR"
)

plt.xlabel("Date")
plt.ylabel("VaR")
plt.title("Comparison of VaR Estimates")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


p = 1 - alpha

lower = int(binom.ppf(beta / 2, J, p))
upper = int(binom.ppf(1 - beta / 2, J, p))

expected = J * p

print("J =", J)
print("Expected exceedances =", expected)
print(f"95% confidence interval = [{lower}, {upper}]")


K_emp = int(results_emp["Exceedance"].sum())
K_ewma = int(results_ewma["Exceedance"].sum())

emp_inside = lower <= K_emp <= upper
ewma_inside = lower <= K_ewma <= upper

print("\nEmpirical method")
print("Observed exceedances =", K_emp)
print("Inside confidence interval =", emp_inside)

print("\nNormal-EWMA method")
print("Observed exceedances =", K_ewma)
print("Inside confidence interval =", ewma_inside)


summary = pd.DataFrame({
    "Method": ["Empirical", "Normal-EWMA"],
    "Exceedances": [K_emp, K_ewma],
    "Exceedance Rate": [K_emp / J, K_ewma / J],
    "Inside 95% CI": [emp_inside, ewma_inside]
})

display(summary)
