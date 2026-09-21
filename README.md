# MF 731 - Corporate Risk Management

Codice per gli HW del corso di Corporate Risk Management.

- `hw1_volatility_theory/` - HW1 (calcoli di volatilità, implied vol, EWMA, GARCH)
- `hw2_spy_volatility/` - HW2 (volatilità di SPY: MA, EWMA, fit e forecast GARCH(1,1))

Ogni cartella contiene lo script `.py` e lo stesso codice come notebook `.ipynb`.

## Come eseguire

```
pip install -r requirements.txt
python hw1_volatility_theory/hw1_volatility_theory.py
python hw2_spy_volatility/hw2_spy_volatility.py
```

HW2 scarica i dati di SPY da Yahoo Finance, quindi serve una connessione internet.
