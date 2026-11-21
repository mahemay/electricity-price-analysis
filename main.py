"""
main.py
-------
End-to-end orchestrator for the Electricity Price Analysis & Forecasting pipeline.
Executes data generation, preprocessing, EDA, and econometric/ML modeling in sequence,
and exports a Markdown-formatted performance table ready for README.md.
"""

from pathlib import Path
import pandas as pd

from src.data_generator import generate_electricity_market_data
from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.modeling import run_modeling


def export_readme_table(
    results_df: pd.DataFrame,
    output_path: str = "outputs/tables/readme_metrics_table.md",
) -> str:
    """Format out-of-sample model metrics as a clean Markdown table for README.md."""
    role_map = {
        "OLS Regression (HAC)": "Causal & Economic Interpretability",
        "XGBoost Regressor": "Non-Linear Forecasting",
    }

    lines = [
        "| Model | Primary Role | RMSE | MAE | $R^2$ |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    for _, row in results_df.iterrows():
        model_name = row["Model"]
        role = role_map.get(model_name, "Forecasting")
        lines.append(
            f"| **{model_name}** | {role} | `{row['RMSE']:.3f}` | "
            f"`{row['MAE']:.3f}` | `{row['R2']:.4f}` |"
        )

    markdown_table = "\n".join(lines)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(markdown_table + "\n", encoding="utf-8")
    return markdown_table


def main() -> None:
    print("=" * 64)
    print("⚡ Electricity Price Analysis & Forecasting — End-to-End Pipeline")
    print("=" * 64)

    # Step 1: Data Generation
    print("\n[Step 1/4] Generating electricity market data...")
    generate_electricity_market_data()

    # Step 2: Data Cleaning & Feature Engineering
    print("\n[Step 2/4] Running preprocessing & feature engineering...")
    run_preprocessing()

    # Step 3: Exploratory Data Analysis (EDA)
    print("\n[Step 3/4] Generating EDA visualizations...")
    run_eda()

    # Step 4: Econometric (OLS) & ML (XGBoost) Modeling
    print("\n[Step 4/4] Training and evaluating OLS & XGBoost models...")
    results_df = run_modeling()

    # Export Markdown table for README.md
    readme_table = export_readme_table(results_df)
    print("\n" + "-" * 64)
    print("📋 Markdown Table for README.md (saved to outputs/tables/readme_metrics_table.md):")
    print("-" * 64)
    print(readme_table)

    print("\n" + "=" * 64)
    print("✅ Pipeline completed successfully! All outputs are ready in outputs/.")
    print("=" * 64)


if __name__ == "__main__":
    main()
