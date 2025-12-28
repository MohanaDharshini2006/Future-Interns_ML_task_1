yes# Retail Sales Forecasting & Predictive Analytics Dashboard

Project scaffold for an end-to-end retail sales forecasting pipeline. Built to clean historical sales, engineer forecasting features, train a Prophet model, and export Power BI–ready CSVs and business insights.

Contents
- `data/` - raw and processed datasets
- `src/` - scripts for cleaning, feature engineering, and forecasting
- `notebooks/eda.ipynb` - exploratory analysis
- `outputs/` - exported forecasts and insights

Quickstart
1. Place your raw CSV into `data/raw/` (columns: `Date`, `Store`, `Region`, `Category`, `Product`, `Sales`).
2. Install dependencies: `pip install pandas numpy prophet scikit-learn matplotlib seaborn`
3. Run cleaning:
```bash
python src/data_cleaning.py data/raw/your_sales.csv --out data/processed/cleaned_monthly.csv
```
4. Create features:
```bash
python src/feature_engineering.py data/processed/cleaned_monthly.csv --out data/processed/fe_monthly.csv
```
5. Forecast (6–12 months):
```bash
python src/forecasting.py data/processed/cleaned_monthly.csv --periods 12 --out outputs/forecasts.csv --insights outputs/insights.csv
```

Outputs for Power BI
- `outputs/forecasts.csv` - `ds,y,yhat,yhat_lower,yhat_upper` (merge for Actual vs Forecast visuals)
- `outputs/insights.csv` - business-ready metrics (best/worst month, YoY growth, ramp-up months)

Business Impact
- Enables forecasting inventory and promotions planning
- Highlights seasonal windows to ramp inventory
- Provides concise metrics for stakeholder dashboards

Contact
For questions or extensions (XGBoost forecasting, hierarchical store forecasts), contact the project owner.
