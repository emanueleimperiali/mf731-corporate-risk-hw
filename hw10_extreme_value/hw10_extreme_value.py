import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, genextreme, genpareto

V = 1_000_000
alphas = np.linspace(0.99, 0.9999, 100)


var_emp = np.quantile(
    losses, alphas, method="inverted_cdf"
)


x = returns.loc[:"2021-11-01"].to_numpy()
lam = theta = 0.97

mu = x[:100].mean()
var = x[:100].var(ddof=1)

for r in x[100:]:
    var = lam * var + (1 - lam) * (r - mu)**2
    mu = theta * mu + (1 - theta) * r

var_normal = -V * mu + V * np.sqrt(var) * norm.ppf(alphas)


losses_gev = losses.iloc[:2555]

def gev_var(returns, n, alphas):
    x = -returns.to_numpy()
    m = len(x) // n

    maxima = x[:m*n].reshape(m, n).max(axis=1)

    c, loc, scale = genextreme.fit(maxima)

    return V * genextreme.ppf(
        alphas**n, c, loc=loc, scale=scale
    )

var_gev_73 = gev_var(returns, 73, alphas)
var_gev_35 = gev_var(returns, 35, alphas)


u = np.quantile(losses, 0.95, method="inverted_cdf")
excesses = losses[losses > u] - u

c, _, scale = genpareto.fit(excesses, floc=0)

p = 1 - (1 - alphas) / 0.05
var_gp = u + genpareto.ppf(p, c, loc=0, scale=scale)


plt.figure(figsize=(9, 5))

plt.plot(alphas, var_emp, label="Empirical")
plt.plot(alphas, var_normal, label="Normal EWMA")
plt.plot(alphas, var_gev_73, label="GEV (n=73)")
plt.plot(alphas, var_gev_35, label="GEV (n=35)")
plt.plot(alphas, var_gp, label="GP")

plt.xlabel("Confidence level")
plt.ylabel("VaR ($)")
plt.title("Daily VaR Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


results = pd.Series({
    "Empirical": var_emp[-1],
    "Normal EWMA": var_normal[-1],
    "GEV (n=73)": var_gev_73[-1],
    "GEV (n=35)": var_gev_35[-1],
    "GP": var_gp[-1]
}).sort_values(ascending=False)

print(results.to_string(float_format=lambda x: f"${x:,.2f}"))
print("\nHighest VaR:", results.idxmax())

print("\nGP vs GEV (n=73):",
      round(100 * (var_gp[-1] / var_gev_73[-1] - 1), 2), "%")

print("GEV block-size difference:",
      round(100 * (var_gev_35[-1] / var_gev_73[-1] - 1), 2), "%")
