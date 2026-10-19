import numpy as np
import itertools

# Parameters
r = 0.0132

sigma1 = 0.2214
sigma2 = 0.3036
ρ = 0.237

T = 0.25
S1 = 158.12
S2 = 170.33

Delta = 1 / 252
horizon = 5 * Delta


def option_price(t, s1, s2, sig1, sig2, corr):
    
    tau = T - t
    
    return np.exp(r * tau) * (
        s1**2 * np.exp(sig1**2 * tau)
        - 2 * s1 * s2 * np.exp(sig1 * sig2 * corr * tau)
        + s2**2 * np.exp(sig2**2 * tau)
    )


def deltas(t, s1, s2, sig1, sig2, corr):
    
    tau = T - t
    
    delta1 = np.exp(r * tau) * (
        2 * s1 * np.exp(sig1**2 * tau)
        - 2 * s2 * np.exp(sig1 * sig2 * corr * tau)
    )
    
    delta2 = np.exp(r * tau) * (
        -2 * s1 * np.exp(sig1 * sig2 * corr * tau)
        + 2 * s2 * np.exp(sig2**2 * tau)
    )
    
    return delta1, delta2


# Initial values
V0_option = option_price(0, S1, S2, sigma1, sigma2, ρ)

h1, h2 = deltas(0, S1, S2, sigma1, sigma2, ρ)

V0_portfolio = V0_option - h1 * S1 - h2 * S2

print("Initial option price:", V0_option)
print("Delta 1:", h1)
print("Delta 2:", h2)
print("Initial portfolio value:", V0_portfolio)


returns = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
vol_mult = [0.50, 0.75, 1.25, 1.50, 1.75, 2.00]
ρ_changes = [-0.50, 0.00, 0.50]

results = []

for x1, x2, m1, m2, dρ in itertools.product(
    returns, returns, vol_mult, vol_mult, ρ_changes
):
    
    # New stock prices
    S1_new = S1 * np.exp(x1)
    S2_new = S2 * np.exp(x2)
    
    # New volatilities
    sigma1_new = sigma1 * m1
    sigma2_new = sigma2 * m2
    
    # New correlation
    ρ_new = ρ + dρ
    
    # New option price
    V_option_new = option_price(
        horizon,
        S1_new,
        S2_new,
        sigma1_new,
        sigma2_new,
        ρ_new
    )
    
    # Portfolio value: deltas are held fixed
    V_portfolio_new = (
        V_option_new
        - h1 * S1_new
        - h2 * S2_new
    )
    
    loss = -(V_portfolio_new - V0_portfolio)
    
    results.append([
        loss, x1, x2, m1, m2,
        dρ, sigma1_new, sigma2_new, ρ_new
    ])


worst = max(results, key=lambda x: x[0])

print("Maximum portfolio loss:", worst[0])

print("\nScenario:")
print("Log return 1:", worst[1])
print("Log return 2:", worst[2])
print("Volatility multiplier 1:", worst[3])
print("Volatility multiplier 2:", worst[4])
print("New volatility 1:", worst[6])
print("New volatility 2:", worst[7])
print("Correlation change:", worst[5])
print("New correlation:", worst[8])


def return_weight(x):
    
    x = round(abs(x), 2)
    
    if x == 0.20:
        return 0.5
    elif x == 0.10:
        return 0.75
    else:
        return 1.0


def volatility_weight(m):
    
    if m in [0.50, 2.00]:
        return 0.5
    elif m in [0.75, 1.75]:
        return 1.25
    else:
        return 0.75


def correlation_weight(dρ):
    
    if dρ == 0:
        return 0.5
    else:
        return 1.25


weighted_results = []

for row in results:
    
    loss, x1, x2, m1, m2, dρ, sig1_new, sig2_new, ρ_new = row
    
    w = (
        return_weight(x1)
        * return_weight(x2)
        * volatility_weight(m1)
        * volatility_weight(m2)
        * correlation_weight(dρ)
    )
    
    weighted_loss = w * loss
    
    weighted_results.append([
        weighted_loss, loss, w,
        x1, x2, m1, m2,
        dρ, sig1_new, sig2_new, ρ_new
    ])


worst_weighted = max(weighted_results, key=lambda x: x[0])

print("Maximum weighted loss:", worst_weighted[0])
print("Actual portfolio loss:", worst_weighted[1])
print("Scenario weight:", worst_weighted[2])

print("\nScenario:")
print("Log return 1:", worst_weighted[3])
print("Log return 2:", worst_weighted[4])
print("Volatility multiplier 1:", worst_weighted[5])
print("Volatility multiplier 2:", worst_weighted[6])
print("New volatility 1:", worst_weighted[8])
print("New volatility 2:", worst_weighted[9])
print("Correlation change:", worst_weighted[7])
print("New correlation:", worst_weighted[10])
