# TimesFM Price Forecaster

Forecast stock, commodity or crypto prices with Google's [TimesFM 3.0](https://github.com/google-research/timesfm).
Uses price history only. **Not financial advice.**

## Run the web app

```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt
python app.py        # open http://localhost:7860
```

The first run downloads the model weights (about 1 GB). It runs on CPU; a forecast takes under a minute.

## Command line

```shell
python forecast.py NVDA          # writes nvda_forecast.csv / .png
python forecast.py GC=F gold     # gold futures
```

TimesFM 3.0 weights are licensed for non-commercial use only.
