"""Data cleaning utilities for retail sales forecasting.

Functions:
- load_raw(csv_path)
- clean_sales(df)
- aggregate_monthly(df, date_col='Date', sales_col='Sales')
- remove_outliers_iqr(df, col)

Save cleaned monthly CSV to data/processed/cleaned_monthly.csv
"""
from typing import Optional
import pandas as pd
import numpy as np
import os


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean_sales(df: pd.DataFrame, date_col: str = "Date", sales_col: str = "Sales") -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Convert date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=[date_col])

    # Ensure sales numeric
    df[sales_col] = pd.to_numeric(df.get(sales_col, df.columns[-1]), errors="coerce")

    # Fill missing sales with 0 for safe aggregation (business decision: missing -> 0)
    df[sales_col] = df[sales_col].fillna(0.0)

    return df


def aggregate_monthly(df: pd.DataFrame, date_col: str = "Date", sales_col: str = "Sales",
                      group_cols: Optional[list] = None) -> pd.DataFrame:
    df = df.copy()
    if group_cols is None:
        group_cols = []

    df["year_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    agg_cols = group_cols + ["year_month"]
    monthly = df.groupby(agg_cols)[sales_col].sum().reset_index().rename(columns={"year_month": "ds", sales_col: "y"})
    return monthly


def remove_outliers_iqr(df: pd.DataFrame, col: str = "y") -> pd.DataFrame:
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = df[col].between(lower, upper)
    return df.loc[mask].reset_index(drop=True)


def save_cleaned(monthly_df: pd.DataFrame, out_path: str = "data/processed/cleaned_monthly.csv") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    monthly_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # quick CLI for local runs
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw retail sales CSV and aggregate monthly.")
    parser.add_argument("input", help="Path to raw CSV")
    parser.add_argument("--out", default="data/processed/cleaned_monthly.csv")
    args = parser.parse_args()

    df_raw = load_raw(args.input)
    df_clean = clean_sales(df_raw)
    monthly = aggregate_monthly(df_clean)
    monthly = remove_outliers_iqr(monthly)
    saved = save_cleaned(monthly, args.out)
    print(f"Saved cleaned monthly data to {saved}")
