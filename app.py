"""Web UI for TimesFM price forecasts.

Run:  python app.py   then open http://localhost:7860
"""
import functools
import os
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from timesfm3 import ModelConfig, TimesFM3Evaluator

PRESETS = {
    "NVIDIA (NVDA)": "NVDA",
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Crude oil (CL=F)": "CL=F",
}
HORIZONS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}
CONTEXT = 1024


@functools.cache
def get_model():
  return TimesFM3Evaluator(ModelConfig(
      checkpoint_path="google/timesfm-3.0-pytorch",
      per_core_batch_size=1, device="cpu"))


def load_prices(ticker):
  df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
  if df.empty:
    raise gr.Error(f"No price data found for '{ticker}'. Check the symbol.")
  return df["Close"].squeeze().dropna()


def run(preset, custom, horizon_label):
  ticker = (custom or "").strip().upper() or PRESETS[preset]
  horizon = HORIZONS[horizon_label]
  close = load_prices(ticker)
  ctx = close.values[-CONTEXT:].astype(np.float32)
  out = list(get_model().predict_batch(
      [ctx], horizon=horizon, return_quantiles=True,
      use_symmetric_averaging=False))[0]

  # Crypto trades every day; everything else on business days.
  freq = "D" if ticker.endswith("-USD") else "B"
  dates = pd.date_range(close.index[-1] + pd.Timedelta(days=1),
                        periods=horizon, freq=freq)
  res = pd.DataFrame({"Median": out.forecast, "Low": out.quantiles[:, 0],
                      "High": out.quantiles[:, 8]}, index=dates)

  hist = close.iloc[-max(250, horizon * 2):]
  fig = go.Figure([
      go.Scatter(x=hist.index, y=hist.values, name="History",
                 line=dict(color="#2563eb")),
      go.Scatter(x=res.index, y=res["High"], line=dict(width=0),
                 showlegend=False, hoverinfo="skip"),
      go.Scatter(x=res.index, y=res["Low"], fill="tonexty",
                 fillcolor="rgba(249,115,22,0.2)", line=dict(width=0),
                 name="10–90% range"),
      go.Scatter(x=res.index, y=res["Median"], name="Forecast (median)",
                 line=dict(color="#f97316")),
  ])
  fig.update_layout(title=f"{ticker} forecast", hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.15))

  last = float(close.iloc[-1])
  end = res.iloc[-1]
  pct = (end["Median"] / last - 1) * 100
  summary = (f"**{ticker}** last close **{last:,.2f}** on {close.index[-1].date()}  \n"
             f"In {horizon_label}: median **{end['Median']:,.2f}** ({pct:+.1f}%), "
             f"likely range {end['Low']:,.2f} – {end['High']:,.2f}")

  step = max(1, horizon // 6)
  table = res.iloc[step - 1::step].map(lambda v: f"{v:,.0f}" if v >= 1000 else f"{v:,.2f}")
  table.insert(0, "Date", table.index.strftime("%b %-d '%y"))
  name = ticker.replace("^", "").replace("=", "_")
  csv_path = os.path.join(tempfile.mkdtemp(), f"{name}_forecast.csv")
  res.astype(float).round(2).to_csv(csv_path, index_label="Date")
  return summary, fig, table, csv_path


with gr.Blocks(title="TimesFM Forecaster") as demo:
  gr.Markdown("# TimesFM Price Forecaster\nForecasts from Google's TimesFM 3.0 "
              "using price history only. **Not financial advice.**")
  with gr.Row():
    preset = gr.Dropdown(list(PRESETS), value="NVIDIA (NVDA)", label="Asset")
    custom = gr.Textbox(label="Or any Yahoo Finance symbol",
                        placeholder="e.g. MSFT, ETH-USD, EURUSD=X")
    horizon = gr.Radio(list(HORIZONS), value="6 months", label="Forecast ahead")
  btn = gr.Button("Forecast", variant="primary")
  summary = gr.Markdown()
  plot = gr.Plot()
  table = gr.Dataframe(label="Forecast checkpoints", interactive=False)
  csv = gr.File(label="Download full forecast (CSV)")
  btn.click(run, [preset, custom, horizon], [summary, plot, table, csv])

if __name__ == "__main__":
  get_model()  # load weights once at startup
  demo.launch(server_name="0.0.0.0", server_port=7860)
