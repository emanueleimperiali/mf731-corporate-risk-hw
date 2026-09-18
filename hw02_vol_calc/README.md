# HW 02 — Volatility Calculations (`02_vol_calc_pub`)

Solves the 5 problems in the assignment:

1. Scaling a 2%/day volatility to a 3-day standard deviation.
2. Implied volatility of a European call option (Black-Scholes, solved with
   Newton-Raphson, falling back to bisection).
3. One-step EWMA volatility update (λ = 0.94).
4. One-step EWMA and GARCH(1,1) volatility updates, compared side by side.
5. GARCH(1,1) long-term volatility, 10-day and 500-day expected volatility
   forecasts, and a short proof that the forecast converges to the
   long-term level as the horizon grows.

## Run

```
python3 vol_calc.py
```

No external dependencies — only the Python standard library (`math`).

## Output

```
Problem 1: sigma_3day = 2% * sqrt(3) = 3.464102%
Problem 2: implied volatility = 14.118359%
Problem 3: updated EWMA volatility = 1.510519%
Problem 4: (a) EWMA = 1.270931%   (b) GARCH(1,1) = 1.275295%
Problem 5: (a) long-term volatility = 1.440392%
           (b) 10-day forecast = 1.715083%, 500-day forecast = 1.452722%
           (c) see convergence argument printed by the script
```
