"""
data_generator.py
-----------------
Generates realistic daily wholesale electricity market data for analysis and forecasting.
Designed to be modularly replaced with real-world API feeds (e.g., ENTSO-E or EIA).
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_electricity_market_data(
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    save_path: str | None = "data/raw/electricity_market_data.csv",
) -> pd.DataFrame:
    """
    Simulate daily wholesale electricity market fundamentals and spot prices.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    seed : int
        Random seed for reproducibility.
    save_path : str or None
        Optional path to save the generated CSV dataset.

    Returns
    -------
    pd.DataFrame
        Simulated daily electricity market dataset.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.to_numpy()
    is_weekend = (dates.dayofweek >= 5).astype(int)

    # 1. Temperature (°C): Seasonal cycle + daily weather shocks
    seasonal_temp = 13.0 - 11.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    temperature = seasonal_temp + rng.normal(0, 3.2, size=n)

    # 2. Natural Gas Price (EUR/MWh): Mean-reverting AR(1) + winter premium
    gas_price = np.zeros(n)
    gas_price[0] = 35.0
    for t in range(1, n):
        winter_premium = 6.0 * np.cos(2 * np.pi * (day_of_year[t] - 15) / 365.25)
        mean_level = 38.0 + winter_premium
        gas_price[t] = (
            gas_price[t - 1]
            + 0.04 * (mean_level - gas_price[t - 1])
            + rng.normal(0, 2.2)
        )
    gas_price = np.clip(gas_price, 12.0, 140.0)

    # 3. Renewable Generation (MWh): Solar (summer peak) + Wind (winter peak)
    solar_gen = np.clip(
        1800 - 1100 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25) + rng.normal(0, 250, size=n),
        200,
        None,
    )
    wind_gen = np.clip(
        2600 + 800 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25) + rng.normal(0, 650, size=n),
        300,
        None,
    )
    renewable_gen = solar_gen + wind_gen

    # 4. Electricity Demand (MWh): Driven by temperature extremes, weekends, and slight growth trend
    hdd_true = np.maximum(0, 18.0 - temperature)
    cdd_true = np.maximum(0, temperature - 22.0)
    trend = np.linspace(0, 450, n)
    demand = (
        11500
        + trend
        + 165 * hdd_true
        + 210 * cdd_true
        - 1400 * is_weekend
        + rng.normal(0, 380, size=n)
    )

    # 5. Wholesale Electricity Price (EUR/MWh):
    # Gas pass-through + demand pressure - renewable Merit Order Effect + AR(1) persistence + spikes
    price = np.zeros(n)
    price[0] = 65.0
    spike_events = rng.binomial(1, 0.025, size=n) * rng.uniform(25, 70, size=n)

    for t in range(1, n):
        fundamental_price = (
            18.0
            + 1.35 * gas_price[t]
            + 0.0048 * demand[t]
            - 0.0065 * renewable_gen[t]
            + spike_events[t]
        )
        # Autoregressive persistence (yesterday's price influences today)
        price[t] = 0.25 * price[t - 1] + 0.75 * fundamental_price + rng.normal(0, 4.5)

    price = np.clip(price, 5.0, None)

    df = pd.DataFrame(
        {
            "date": dates,
            "electricity_price": np.round(price, 2),
            "demand_mwh": np.round(demand, 1),
            "gas_price": np.round(gas_price, 2),
            "renewable_gen_mwh": np.round(renewable_gen, 1),
            "temperature_c": np.round(temperature, 2),
        }
    )

    # Inject ~1% realistic missing values to demonstrate cleaning in preprocessing.py
    for col in ["demand_mwh", "gas_price", "renewable_gen_mwh", "temperature_c"]:
        missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    if save_path:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"[data_generator] Dataset saved to {output_file} ({len(df)} rows).")

    return df


if __name__ == "__main__":
    generate_electricity_market_data()
