"""
MF 731 - Corporate Risk Management
Homework 02 - Volatility calculations (02_vol_calc_pub)

Solves problems 1-5:
  1. Scaling daily volatility to a 3-day horizon.
  2. Implied volatility of a European call (Black-Scholes, Newton-Raphson).
  3. One-step EWMA volatility update.
  4. One-step EWMA and GARCH(1,1) volatility updates.
  5. GARCH(1,1) long-term volatility and multi-step variance forecasts.

Only the Python standard library is used (math), so the script runs anywhere
with no extra dependencies.
"""

import math


# ---------------------------------------------------------------------------
# Problem 1: volatility scaling with the square-root-of-time rule
# ---------------------------------------------------------------------------
def problem1(daily_vol: float, n_days: int) -> float:
    """Std dev of the percentage price change over n_days, from a daily vol."""
    return daily_vol * math.sqrt(n_days)


# ---------------------------------------------------------------------------
# Problem 2: implied volatility via Black-Scholes + Newton-Raphson
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_vega(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S0 * math.sqrt(T) * _norm_pdf(d1)


def implied_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    sigma0: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson solve for sigma such that BS price == market_price,
    with a bisection fallback in case Newton steps leave sigma non-positive
    or fail to converge."""
    sigma = sigma0
    for _ in range(max_iter):
        price = bs_call_price(S0, K, T, r, sigma)
        vega = bs_vega(S0, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-12:
            break
        step = diff / vega
        new_sigma = sigma - step
        if new_sigma <= 0:
            break
        sigma = new_sigma
    else:
        return sigma

    # Bisection fallback on a wide bracket.
    lo, hi = 1e-6, 5.0
    f_lo = bs_call_price(S0, K, T, r, lo) - market_price
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_price(S0, K, T, r, mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Problems 3-4: EWMA and GARCH(1,1) one-step variance updates
# ---------------------------------------------------------------------------
def ewma_update(sigma_prev: float, return_today: float, lam: float) -> float:
    """sigma_prev, return_today are per-period (e.g. daily) decimals."""
    var_new = lam * sigma_prev ** 2 + (1 - lam) * return_today ** 2
    return math.sqrt(var_new)


def garch11_update(
    sigma_prev: float, return_today: float, alpha0: float, alpha1: float, beta1: float
) -> float:
    var_new = alpha0 + alpha1 * return_today ** 2 + beta1 * sigma_prev ** 2
    return math.sqrt(var_new)


def pct_return(price_yesterday: float, price_today: float) -> float:
    return (price_today - price_yesterday) / price_yesterday


# ---------------------------------------------------------------------------
# Problem 5: GARCH(1,1) long-term volatility and n-step-ahead forecasts
# ---------------------------------------------------------------------------
def garch11_long_run_variance(alpha0: float, alpha1: float, beta1: float) -> float:
    return alpha0 / (1 - alpha1 - beta1)


def garch11_forecast_variance(
    sigma_t: float, alpha1: float, beta1: float, VL: float, n: int
) -> float:
    """E[sigma_{t+n}^2] = VL + (alpha1+beta1)^n * (sigma_t^2 - VL)."""
    persistence = alpha1 + beta1
    return VL + (persistence ** n) * (sigma_t ** 2 - VL)


def print_convergence_argument(alpha1: float, beta1: float) -> None:
    persistence = alpha1 + beta1
    print(
        "  Proof sketch: E[sigma_{t+n}^2] = VL + (alpha1+beta1)^n * (sigma_t^2 - VL).\n"
        f"  Since alpha1 + beta1 = {persistence:.6f} < 1 (stationarity condition),\n"
        "  (alpha1+beta1)^n -> 0 as n -> infinity, so E[sigma_{t+n}^2] -> VL,\n"
        "  i.e. the expected variance (and hence volatility) converges to the\n"
        "  long-term level as the forecast horizon grows."
    )


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Problem 1: 3-day standard deviation from 2%/day volatility")
    print("=" * 70)
    sd_3day = problem1(0.02, 3)
    print(f"  sigma_3day = 2% * sqrt(3) = {sd_3day:.6%}\n")

    print("=" * 70)
    print("Problem 2: Implied volatility (Black-Scholes)")
    print("=" * 70)
    S0, K, T, r, C_market = 120.0, 115.0, 0.5, 0.03, 8.75
    iv = implied_vol(S0, K, T, r, C_market)
    check_price = bs_call_price(S0, K, T, r, iv)
    print(f"  S0={S0}, K={K}, T={T}, r={r}, C_market={C_market}")
    print(f"  implied volatility = {iv:.6%}")
    print(f"  BS price at that vol (check) = {check_price:.6f}\n")

    print("=" * 70)
    print("Problem 3: EWMA update (lambda = 0.94)")
    print("=" * 70)
    sigma_prev, lam = 0.015, 0.94
    p_yst, p_today = 30.00, 30.50
    u = pct_return(p_yst, p_today)
    sigma_new = ewma_update(sigma_prev, u, lam)
    print(f"  return today u = {u:.6%}")
    print(f"  updated EWMA volatility = {sigma_new:.6%}\n")

    print("=" * 70)
    print("Problem 4: EWMA vs GARCH(1,1) update")
    print("=" * 70)
    sigma_prev = 0.013
    p_yst, p_today = 300.0, 298.0
    u = pct_return(p_yst, p_today)
    print(f"  return today u = {u:.6%}")

    sigma_ewma = ewma_update(sigma_prev, u, 0.94)
    print(f"  (a) EWMA (lambda=0.94):   updated volatility = {sigma_ewma:.6%}")

    sigma_garch = garch11_update(sigma_prev, u, 0.000002, 0.04, 0.94)
    print(f"  (b) GARCH(1,1):           updated volatility = {sigma_garch:.6%}\n")

    print("=" * 70)
    print("Problem 5: GARCH(1,1) long-term volatility and forecasts")
    print("=" * 70)
    alpha0, alpha1, beta1 = 0.0000013465, 0.083394, 0.910116
    VL = garch11_long_run_variance(alpha0, alpha1, beta1)
    vol_L = math.sqrt(VL)
    print(f"  (a) long-term daily volatility sqrt(VL) = {vol_L:.6%}")

    sigma_t = 0.01732
    for n in (10, 500):
        var_n = garch11_forecast_variance(sigma_t, alpha1, beta1, VL, n)
        print(f"  (b) E[volatility] after {n:>3} days = {math.sqrt(var_n):.6%}")

    print("  (c)")
    print_convergence_argument(alpha1, beta1)


if __name__ == "__main__":
    main()
