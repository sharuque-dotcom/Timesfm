"""Forecast a ticker's daily close with TimesFM 3.0 (CPU).

Usage: python forecast.py [TICKER] [LABEL]   e.g. python forecast.py GC=F gold
"""
import sys, numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timesfm3 import TimesFM3Evaluator, ModelConfig

TICKER = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
LABEL = sys.argv[2] if len(sys.argv) > 2 else TICKER.lower()
HORIZON = 126  # ~6 months of trading days
CONTEXT = 1024

df = yf.download(TICKER, period="10y", auto_adjust=True, progress=False)
close = df["Close"].squeeze().dropna()
ctx = close.values[-CONTEXT:].astype(np.float32)
print("Last close:", close.index[-1].date(), round(float(close.iloc[-1]), 2))

model = TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                      per_core_batch_size=1, device="cpu"))
out = list(model.predict_batch([ctx], horizon=HORIZON, return_quantiles=True,
                               use_symmetric_averaging=False))[0]
q = out.quantiles  # (H, 9): deciles 0.1..0.9
dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
res = pd.DataFrame({"median": out.forecast, "p10": q[:, 0], "p90": q[:, 8]}, index=dates)
res.round(2).to_csv(f"{LABEL}_forecast.csv")
for n in [5, 21, 42, 63, 126]:
    r = res.iloc[n - 1]; print(f"{dates[n-1].date()}  median {r['median']:.2f}  10-90%: {r.p10:.2f}-{r.p90:.2f}")

hist = close.iloc[-250:]
plt.figure(figsize=(10, 5))
plt.plot(hist.index, hist.values, label=f"{TICKER} close (last year)")
plt.plot(res.index, res["median"], label="TimesFM median")
plt.fill_between(res.index, res.p10, res.p90, alpha=.25, label="10–90% range")
plt.title(f"{TICKER} – TimesFM 3.0 forecast (not financial advice)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{LABEL}_forecast.png", dpi=150)
