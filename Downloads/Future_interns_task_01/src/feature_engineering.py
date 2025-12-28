"""Feature engineering for time series forecasting.

Creates month/year, rolling averages, lag features, seasonal indicators and holiday flag.
"""
from typing import Optional, List
import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    df = df.copy()
    df[ds_col] = pd.to_datetime(df[ds_col])
    df["month"] = df[ds_col].dt.month
    df["month_name"] = df[ds_col].dt.month_name()
    df["year"] = df[ds_col].dt.year
    df["quarter"] = df[ds_col].dt.quarter
    return df


def add_rolling_and_lags(df: pd.DataFrame, col: str = "y", windows: List[int] = [3, 6], lags: List[int] = [1, 3, 6]) -> pd.DataFrame:
    df = df.sort_values("ds").reset_index(drop=True)
    for w in windows:
        df[f"rolling_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    for l in lags:
        df[f"lag_{l}"] = df[col].shift(l)
    # backfill lag NaNs with zeros (business choice) to keep model input stable
    lag_cols = [f"lag_{l}" for l in lags]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def add_seasonal_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # One-hot months
    month_dummies = pd.get_dummies(df["month"].astype(int), prefix="m")
    df = pd.concat([df, month_dummies], axis=1)
    return df


def add_holiday_flag(df: pd.DataFrame, holiday_dates: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
    df = df.copy()
    if holiday_dates is None:
        df["is_holiday"] = 0
    else:
        holiday_set = set(pd.to_datetime(holiday_dates))
        df["is_holiday"] = df["ds"].isin(holiday_set).astype(int)
    return df


def pipeline(features_df: pd.DataFrame, holiday_dates: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
    df = create_time_features(features_df)
    df = add_rolling_and_lags(df)
    df = add_seasonal_indicators(df)
    df = add_holiday_flag(df, holiday_dates=holiday_dates)
    return df


if __name__ == "__main__":
    # example CLI usage
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="data/processed/fe_monthly.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = pipeline(df)
    df.to_csv(args.out, index=False)
    print(f"Saved features to {args.out}")
