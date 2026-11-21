"""
eda.py
------
Generates publication-quality exploratory data analysis (EDA) visualizations:
1. Electricity price trends & volatility clustering
2. Market fundamentals correlation heatmap
3. Merit Order Effect (Renewable penetration vs. Wholesale electricity price)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def _configure_plot_style() -> None:
    """Apply a clean, professional visual theme across all figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.9,
        }
    )


def plot_price_and_volatility(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Plot daily wholesale electricity prices with 30-day moving average
    and rolling volatility clustering.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top panel: Spot price & 30-day moving average
    rolling_30d_mean = df["electricity_price"].rolling(window=30).mean()
    ax1.plot(
        df["date"],
        df["electricity_price"],
        color="#1f77b4",
        alpha=0.55,
        linewidth=0.9,
        label="Daily Spot Price",
    )
    ax1.plot(
        df["date"],
        rolling_30d_mean,
        color="#d62728",
        linewidth=2.0,
        label="30-Day Moving Average",
    )
    ax1.set_title("Wholesale Electricity Price Dynamics & Volatility Clustering")
    ax1.set_ylabel("Price (EUR/MWh)")
    ax1.legend(loc="upper left")

    # Bottom panel: 7-day & 30-day rolling volatility
    ax2.plot(
        df["date"],
        df["price_rolling_std_7"],
        color="#ff7f0e",
        alpha=0.7,
        linewidth=1.0,
        label="7-Day Rolling Volatility (Std)",
    )
    ax2.plot(
        df["date"],
        df["price_rolling_std_30"],
        color="#2ca02c",
        linewidth=1.8,
        label="30-Day Rolling Volatility (Std)",
    )
    ax2.set_ylabel("Volatility (EUR/MWh)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    save_path = output_dir / "price_trend_volatility.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Plot a lower-triangle correlation heatmap of core electricity market variables.
    """
    cols = {
        "electricity_price": "Electricity Price",
        "gas_price": "Natural Gas Price",
        "demand_mwh": "Total Demand",
        "net_load_mwh": "Net Load",
        "renewable_gen_mwh": "Renewable Gen",
        "renewable_share": "Renewable Share",
        "hdd": "HDD (Heating)",
        "cdd": "CDD (Cooling)",
        "price_rolling_std_7": "7D Volatility",
    }

    corr_df = df[list(cols.keys())].rename(columns=cols).corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Electricity Market Fundamentals")
    plt.tight_layout()

    save_path = output_dir / "correlation_heatmap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_merit_order_effect(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Visualize the Merit Order Effect: relationship between renewable energy share
    and wholesale electricity prices, color-coded by natural gas price regimes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    renewable_pct = df["renewable_share"] * 100.0
    scatter = ax.scatter(
        renewable_pct,
        df["electricity_price"],
        c=df["gas_price"],
        cmap="viridis",
        alpha=0.65,
        edgecolors="none",
        s=28,
    )

    # Fit and plot linear trendline to highlight downward price pressure
    slope, intercept = np.polyfit(renewable_pct, df["electricity_price"], 1)
    x_line = np.linspace(renewable_pct.min(), renewable_pct.max(), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=f"Linear Trend (Slope = {slope:.2f} EUR/MWh per +1% RE)",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Natural Gas Price (EUR/MWh)")

    ax.set_title("Merit Order Effect: Renewable Penetration vs. Electricity Price")
    ax.set_xlabel("Renewable Energy Share of Demand (%)")
    ax.set_ylabel("Wholesale Electricity Price (EUR/MWh)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    save_path = output_dir / "merit_order_effect.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_eda(
    processed_path: str = "data/processed/electricity_market_processed.csv",
    figures_dir: str = "outputs/figures",
) -> list[Path]:
    """
    Execute all EDA visualizations and save figures to disk.
    """
    data_file = Path(processed_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_file}. Run src/preprocessing.py first."
        )

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_plot_style()
    df = pd.read_csv(data_file, parse_dates=["date"])

    saved_paths = [
        plot_price_and_volatility(df, output_dir),
        plot_correlation_heatmap(df, output_dir),
        plot_merit_order_effect(df, output_dir),
    ]

    for path in saved_paths:
        print(f"[eda] Saved figure: {path}")

    return saved_paths


if __name__ == "__main__":
    run_eda()
