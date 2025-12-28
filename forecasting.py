"""Forecasting utilities using Prophet with fallbacks.

Exports forecasts to outputs/forecasts.csv and insights to outputs/insights.csv
"""
from typing import Optional
import pandas as pd
import os
import numpy as np

try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None


def train_and_forecast(df: pd.DataFrame, periods: int = 12, freq: str = "M") -> pd.DataFrame:
    df = df.copy()
    # Expect ds,y
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("Input df must contain 'ds' and 'y' columns")

    if Prophet is None:
        raise ImportError("Prophet is not installed. Install 'prophet' or 'fbprophet' to use this module.")

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df.rename(columns={"ds": "ds", "y": "y"}))

    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    # merge historical y if available
    merged = pd.merge(out, df[["ds", "y"]], on="ds", how="left")
    return merged


def save_forecast(df: pd.DataFrame, out_path: str = "outputs/forecasts.csv") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def generate_insights(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, out_path: str = "outputs/insights.csv") -> str:
    # hist_df: historical monthly ds,y
    hist = hist_df.copy()
    hist["month"] = pd.to_datetime(hist["ds"]).dt.month
    monthly_avg = hist.groupby("month")["y"].mean()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()

    # YoY growth: sum last 12 months vs previous 12 months if available
    hist = hist.sort_values("ds")
    if len(hist) >= 24:
        last_12 = hist.tail(12)["y"].sum()
        prev_12 = hist.tail(24).head(12)["y"].sum()
        yoy = (last_12 - prev_12) / prev_12 * 100 if prev_12 != 0 else np.nan
    else:
        yoy = np.nan

    # Forecast peak
    forecast_df = forecast_df.copy()
    forecast_df["month"] = pd.to_datetime(forecast_df["ds"]).dt.month
    peak_month = int(forecast_df.loc[forecast_df["yhat"].idxmax(), "month"]) if not forecast_df.empty else None

    # Recommend ramp-up months: top 3 forecast months by average yhat
    ramp = (forecast_df.groupby("month")["yhat"].mean().sort_values(ascending=False).head(3).index.tolist())

    insights = pd.DataFrame({
        "metric": [
            "best_month",
            "worst_month",
            "yoy_growth_pct",
            "forecasted_peak_month",
            "recommended_inventory_ramp_months"
        ],
        "value": [
            int(best_month),
            int(worst_month),
            round(float(yoy), 2) if not np.isnan(yoy) else "NA",
            int(peak_month) if peak_month is not None else "NA",
            ",".join([str(int(m)) for m in ramp])
        ]
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    insights.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to cleaned monthly CSV (ds,y)")
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--out", default="outputs/forecasts.csv")
    parser.add_argument("--insights", default="outputs/insights.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    fc = train_and_forecast(df, periods=args.periods)
    save_forecast(fc, args.out)
    generate_insights(df, fc, args.insights)
    print(f"Forecast saved to {args.out} and insights saved to {args.insights}")
