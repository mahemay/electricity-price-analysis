# Electricity Price Analysis & Forecasting

An end-to-end data science and energy economics project analyzing electricity price trends, volatility, and key market drivers—including demand, temperature, natural gas prices, and renewable energy generation.

## Project Objectives
- **Data Cleaning & Feature Engineering:** Handling missing values, constructing Heating/Cooling Degree Days (HDD/CDD), rolling volatility, and lag features.
- **Exploratory Data Analysis (EDA):** Visualizing price trends, volatility clusters, correlations, and the Merit Order Effect.
- **Econometric & ML Modeling:** Combining OLS regression (for causal/economic interpretability) with XGBoost (for out-of-sample time-series forecasting).
- **Economic Interpretation:** Translating statistical coefficients into actionable energy market insights.

## Repository Structure
- `src/data_generator.py`: Generates realistic daily electricity market data (replaceable with ENTSO-E or EIA datasets).
- `src/preprocessing.py`: Cleans raw data and engineers domain-specific features.
- `src/eda.py`: Produces time-series, volatility, and correlation visualizations.
- `src/modeling.py`: Trains OLS and XGBoost models using a time-based train/test split.
- `main.py`: Runs the complete pipeline end-to-end.

## Quick Start
```bash
git clone [https://github.com/mahemay/electricity-price-analysis.git](https://github.com/mahemay/electricity-price-analysis.git)
cd electricity-price-analysis
pip install -r requirements.txt
python main.py
