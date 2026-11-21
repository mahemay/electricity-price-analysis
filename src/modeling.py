"""
modeling.py
-----------
Trains and evaluates econometric (OLS) and machine learning (XGBoost) models
for wholesale electricity price analysis and forecasting using a strict
time-based train/test split.

Outputs:
- outputs/tables/model_comparison.csv
- outputs/tables/ols_summary.txt
- outputs/figures/forecast_vs_actual.png
- outputs/figures/feature_importance.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


OLS_FEATURES = [
    "gas_price",
    "demand_mwh",
    "renewable_gen_mwh",
    "hdd",
    "cdd",
    "is_weekend",
    "price_lag_1",
    "price_rolling_std_7",
]

XGB_FEATURES = [
    "gas_price",
    "demand_mwh",
    "renewable_gen_mwh",
    "renewable_share",
    "net_load_mwh",
    "temperature_c",
    "hdd",
    "cdd",
    "day_of_week",
    "month",
    "is_weekend",
    "is_winter",
    "is_summer",
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "demand_lag_1",
    "gas_price_lag_1",
    "price_rolling_mean_7",
    "price_rolling_std_7",
    "price_rolling_std_30",
]

TARGET_COL = "electricity_price"


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split the dataset into train and out-of-sample test sets."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression forecasting metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}


def train_ols(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """
    Fit an OLS regression with Newey-West (HAC) robust standard errors
    to account for autocorrelation and heteroskedasticity in daily prices.
    """
    X_train = sm.add_constant(train_df[OLS_FEATURES])
    y_train = train_df[TARGET_COL]
    X_test = sm.add_constant(test_df[OLS_FEATURES])

    ols_model = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    y_pred = ols_model.predict(X_test).to_numpy()
    return ols_model, y_pred


def train_xgboost(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42
) -> tuple[XGBRegressor, np.ndarray]:
    """Fit a regularized XGBoost Regressor for non-linear time-series forecasting."""
    X_train = train_df[XGB_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[XGB_FEATURES]

    xgb_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    return xgb_model, y_pred


def plot_forecast_vs_actual(
    test_df: pd.DataFrame,
    ols_pred: np.ndarray,
    xgb_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot out-of-sample actual electricity prices vs. OLS and XGBoost forecasts."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    ax.plot(
        test_df["date"],
        test_df[TARGET_COL],
        color="#222222",
        linewidth=1.5,
        alpha=0.85,
        label="Actual Spot Price",
    )
    ax.plot(
        test_df["date"],
        ols_pred,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="OLS Forecast",
    )
    ax.plot(
        test_df["date"],
        xgb_pred,
        color="#d62728",
        linewidth=1.4,
        alpha=0.9,
        label="XGBoost Forecast",
    )

    ax.set_title(
        "Out-of-Sample Electricity Price Forecasting: Actual vs. OLS & XGBoost",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Wholesale Electricity Price (EUR/MWh)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    save_path = output_dir / "forecast_vs_actual.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


def plot_feature_importance(xgb_model: XGBRegressor, output_dir: Path) -> Path:
    """Plot top 12 feature importances from the trained XGBoost model."""
    sns.set_theme(style="whitegrid", context="notebook")
    importances = (
        pd.Series(xgb_model.feature_importances_, index=XGB_FEATURES)
        .sort_values(ascending=True)
        .tail(12)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    importances.plot(kind="barh", color="#2a9d8f", ax=ax)
    ax.set_title(
        "Top 12 Drivers of Electricity Prices (XGBoost Feature Importance)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Relative Importance (Gain)")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    save_path = output_dir / "feature_importance.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


def run_modeling(
    processed_path: str = "data/processed/electricity_market_processed.csv",
    figures_dir: str = "outputs/figures",
    tables_dir: str = "outputs/tables",
) -> pd.DataFrame:
    """Execute the full modeling, evaluation, and reporting pipeline."""
    data_file = Path(processed_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_file}. Run src/preprocessing.py first."
        )

    fig_dir = Path(figures_dir)
    tab_dir = Path(tables_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, parse_dates=["date"])
    train_df, test_df = time_based_split(df, train_ratio=0.8)
    print(
        f"[modeling] Train period: {train_df['date'].min().date()} to {train_df['date'].max().date()} "
        f"({len(train_df)} days)"
    )
    print(
        f"[modeling] Test period:  {test_df['date'].min().date()} to {test_df['date'].max().date()} "
        f"({len(test_df)} days)"
    )

    # 1. Train Models
    ols_model, ols_pred = train_ols(train_df, test_df)
    xgb_model, xgb_pred = train_xgboost(train_df, test_df)

    # 2. Evaluate Out-of-Sample Performance
    y_test = test_df[TARGET_COL].to_numpy()
    results_df = pd.DataFrame(
        [
            {"Model": "OLS Regression (HAC)", **evaluate_predictions(y_test, ols_pred)},
            {"Model": "XGBoost Regressor", **evaluate_predictions(y_test, xgb_pred)},
        ]
    )

    # 3. Save Tables
    metrics_path = tab_dir / "model_comparison.csv"
    results_df.to_csv(metrics_path, index=False)

    ols_summary_path = tab_dir / "ols_summary.txt"
    ols_summary_path.write_text(ols_model.summary().as_text(), encoding="utf-8")

    # 4. Save Figures
    forecast_fig = plot_forecast_vs_actual(test_df, ols_pred, xgb_pred, fig_dir)
    importance_fig = plot_feature_importance(xgb_model, fig_dir)

    print("\n[modeling] Out-of-Sample Model Comparison:")
    print(results_df.to_string(index=False))
    print(f"\n[modeling] Saved metrics table: {metrics_path}")
    print(f"[modeling] Saved OLS summary:   {ols_summary_path}")
    print(f"[modeling] Saved figure:        {forecast_fig}")
    print(f"[modeling] Saved figure:        {importance_fig}")

    return results_df


if __name__ == "__main__":
    run_modeling()
