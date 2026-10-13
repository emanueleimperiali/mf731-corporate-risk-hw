# MF 731 - Corporate Risk Management

Codice per gli HW del corso di Corporate Risk Management.

- `hw1_volatility_theory/` - HW1 (calcoli di volatilità, implied vol, EWMA, GARCH)
- `hw2_spy_volatility/` - HW2 (volatilità di SPY: MA, EWMA, fit e forecast GARCH(1,1))
- `hw3_loss_distribution/` - HW3 (loss distribution di una put: full vs. linearizzata vs. second-order, Monte Carlo)
- `hw4_var_estimation/` - HW4 (VaR di un portafoglio MSFT/AAPL/GOOGL: empirico vs. parametrico con EWMA)

Ogni cartella contiene lo script `.py` e lo stesso codice come notebook `.ipynb`.

## Come eseguire

```
pip install -r requirements.txt
python hw1_volatility_theory/hw1_volatility_theory.py
python hw2_spy_volatility/hw2_spy_volatility.py
python hw3_loss_distribution/hw3_loss_distribution.py
python hw4_var_estimation/hw4_var_estimation.py
```

HW2 e HW4 scaricano i dati da Yahoo Finance, quindi serve una connessione internet.
