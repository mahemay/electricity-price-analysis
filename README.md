# ⚡ Electricity Price Analysis & Forecasting

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Econometrics](https://img.shields.io/badge/Econometrics-OLS%20%7C%20Time%20Series-orange?style=flat)](https://www.statsmodels.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-2ECC71?style=flat)](https://xgboost.readthedocs.io/)
[![Domain](https://img.shields.io/badge/Domain-Energy%20Economics-1E8449?style=flat)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)](LICENSE)

An end-to-end **energy economics and quantitative modeling** pipeline investigating wholesale electricity price dynamics, volatility clustering, and fundamental market drivers—including load demand, weather extremes, natural gas prices, and renewable energy penetration.

---

## 📑 Table of Contents
- [Executive Summary](#-executive-summary)
- [Analytical Pipeline & Methodology](#-analytical-pipeline--methodology)
- [Key Findings & Model Performance](#-key-findings--model-performance)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Extensibility & Policy Applications](#-extensibility--policy-applications)
- [Author](#-author)

---

## 🎯 Executive Summary

Wholesale electricity markets are characterized by **non-storability**, **inelastic short-run demand**, and **pronounced price spikes** driven by fuel costs, weather shocks, and intermittent renewable supply. 

This project bridges the gap between **structural economic interpretability** and **out-of-sample forecasting accuracy** by combining classical econometrics (**OLS Regression**) with non-linear gradient-boosted trees (**XGBoost**). Designed with modularity in mind, the pipeline serves both as an analytical framework for understanding power market fundamentals and as a reproducible forecasting tool ready for live market feeds (**ENTSO-E**, **EIA**).

### Core Analytical Pillars
* **Energy-Domain Feature Engineering:** Constructs Heating and Cooling Degree Days (**HDD/CDD**), rolling price volatility windows, calendar seasonality, and autoregressive lag structures.
* **Market Mechanics & EDA:** Visualizes wholesale price regimes, volatility clustering, gas-to-power price pass-through, and the **Merit Order Effect** of renewable generation.
* **Leakage-Free Evaluation:** Enforces a strict **chronological (time-based) train/test split** to prevent look-ahead bias and mirror real-world operational forecasting.
* **Economic & Policy Translation:** Converts statistical estimates and feature importances into actionable insights for energy market participants and climate policy analysts.

---

## 🔬 Analytical Pipeline & Methodology

### Pipeline Architecture
```text
[src/data_generator.py] ──▶ [src/preprocessing.py] ──▶ [src/eda.py] ──▶ Market Diagnostics & Figures
  (Synthetic / ENTSO-E)      (Cleaning, HDD/CDD,        │
                              Lags, Volatility)         ▼
                                                  [src/modeling.py] ──▶ OLS (Interpretability)
                                                  (Time-Based Split)    XGBoost (Forecasting)
```

### Econometric Specification (Baseline Structural Model)
To isolate marginal economic effects on daily wholesale electricity prices ($P_t$), the baseline OLS specification models price dynamics as a function of market fundamentals and lagged dynamics:

$$P_t = \beta_0 + \beta_1 \text{Demand}_t + \beta_2 \text{GasPrice}_t + \beta_3 \text{Renewables}_t + \beta_4 \text{HDD}_t + \beta_5 \text{CDD}_t + \sum_{k=1}^{p} \phi_k P_{t-k} + \varepsilon_t$$

* **Expected Merit Order Effect ($\beta_3 < 0$):** Higher zero-marginal-cost renewable generation shifts the supply stack rightward, displacing high-marginal-cost thermal units and lowering clearing prices.
* **Gas-to-Power Pass-Through ($\beta_2 > 0$):** Natural gas frequently acts as the marginal price-setting fuel, transmitting fuel market shocks directly into power prices.

---

## 📊 Key Findings & Model Performance

> *Note: Run `python main.py` and replace the bracketed placeholders below with your exact test-set metrics and generated figure paths.*

### 1. Out-of-Sample Forecasting Comparison
| Model | Primary Objective | RMSE | MAE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **OLS Regression** | Structural & Economic Interpretability | `[e.g., 8.42]` | `[e.g., 6.15]` | `[e.g., 0.81]` |
| **XGBoost Regressor** | Non-Linear Predictive Accuracy | `[e.g., 4.19]` | `[e.g., 3.05]` | `[e.g., 0.94]` |

### 2. Core Economic Insights
* **Gas-to-Power Linkage:** Natural gas prices emerge as a primary marginal driver of wholesale electricity prices, exhibiting a positive and statistically significant pass-through.
* **Merit Order Effect:** Increased renewable energy penetration systematically suppresses wholesale clearing prices while altering intraday/daily volatility patterns.
* **Non-Linear Weather Shocks:** Temperature extremes captured via **HDD/CDD** trigger steep demand surges, driving non-linear price spikes better captured by tree-based ensembles (XGBoost) than linear models.

### 3. Visual Diagnostics
*(Upload key figures generated by `src/eda.py` and `src/modeling.py` into `outputs/figures/` to render them automatically below.)*

```markdown
![Forecast vs Actual](outputs/figures/forecast_vs_actual.png)
```
*Figure 1: Out-of-sample wholesale electricity price forecasting performance (Actual vs. Predicted).*

---

## 📁 Repository Structure

```text
electricity-price-analysis/
├── src/
│   ├── data_generator.py    # Simulates realistic daily electricity market data (swappable with ENTSO-E / EIA)
│   ├── preprocessing.py     # Missing-value imputation, HDD/CDD engineering, rolling volatility & lags
│   ├── eda.py               # Time-series trends, volatility clusters, correlation & Merit Order plots
│   └── modeling.py          # Chronological train/test split, OLS & XGBoost training and evaluation
├── outputs/
│   └── figures/             # High-resolution charts exported by EDA and modeling modules
├── main.py                  # End-to-end pipeline orchestrator
├── requirements.txt         # Reproducible Python environment dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mahemay/electricity-price-analysis.git
cd electricity-price-analysis
```

### 2. Set up the environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execute the full pipeline
```bash
python main.py
```

---

## 🔄 Extensibility & Policy Applications

This framework is built to scale into deeper empirical energy research:
* **Live Market Data Integration:** Swap `src/data_generator.py` with live API connectors for **ENTSO-E Transparency Platform** (European wholesale markets) or **U.S. EIA**.
* **Climate Policy & Carbon Pricing:** Incorporate **EU ETS carbon allowance prices** to quantify carbon pass-through rates alongside gas and renewable effects.
* **Volatility & Risk Modeling:** Extend the econometric layer with **GARCH / EGARCH** specifications and **Quantile XGBoost** for extreme price spike and Value-at-Risk (VaR) analysis.

---

## 👤 Author

**Mohammad Akbarzadeh**  
*Energy Economist | Renewable Energy, Climate Policy & Electricity Markets | Data Analysis & Energy Modelling*

[![GitHub](https://img.shields.io/badge/GitHub-mahemay-181717?style=flat&logo=github)](https://github.com/mahemay)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:your.email@example.com)

---

*If you find this project useful for energy economics research or power market modeling, feel free to star ⭐ the repository or open an issue for collaboration.*
