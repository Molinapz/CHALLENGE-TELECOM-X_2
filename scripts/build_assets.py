from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telecomx_churn.pipeline import (  # noqa: E402
    compare_models,
    extract_and_clean_data,
    get_categorical_rate_table,
    get_feature_importances,
    get_numeric_summary_by_churn,
    get_preprocessing_summary,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_correlations,
    plot_directed_analysis,
    plot_feature_importance,
    prepare_modeling_dataset,
    set_plot_theme,
    split_dataset,
)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    set_plot_theme()

    source_repo_json = REPO_ROOT.parent / "TELECOMX_LATAM-CHALLENGE-remote" / "TelecomX_Data.json"
    raw_data_dir = REPO_ROOT / "data" / "raw"
    processed_data_dir = REPO_ROOT / "data" / "processed"
    reports_dir = REPO_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = raw_data_dir / "TelecomX_Data.json"
    shutil.copy2(source_repo_json, raw_data_path)

    clean_df = extract_and_clean_data(raw_data_path)
    clean_df.to_csv(processed_data_dir / "telecomx_clean.csv", index=False)

    preprocessing_summary = get_preprocessing_summary(clean_df)
    preprocessing_summary.to_csv(reports_dir / "class_distribution.csv", index=False)

    get_categorical_rate_table(clean_df, "contract").to_csv(reports_dir / "contract_churn_rates.csv", index=False)
    get_categorical_rate_table(clean_df, "payment_method").to_csv(
        reports_dir / "payment_method_churn_rates.csv",
        index=False,
    )
    get_categorical_rate_table(clean_df, "internet_service").to_csv(
        reports_dir / "internet_service_churn_rates.csv",
        index=False,
    )
    get_numeric_summary_by_churn(clean_df).to_csv(reports_dir / "numeric_summary_by_churn.csv", index=False)

    X, y = prepare_modeling_dataset(clean_df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

    metrics_df, artifacts = compare_models(X_train, X_test, y_train, y_test)
    metrics_df.to_csv(reports_dir / "model_metrics.csv", index=False)

    logistic_coefficients, random_forest_importances = get_feature_importances(artifacts, top_n=12)
    logistic_coefficients.rename("coefficient").to_csv(reports_dir / "logistic_coefficients.csv")
    random_forest_importances.rename("importance").to_csv(reports_dir / "random_forest_feature_importance.csv")

    correlation_figure, target_correlations = plot_correlations(clean_df)
    target_correlations.round(4).rename("correlation").to_csv(reports_dir / "target_correlations.csv")

    save_figure(plot_class_distribution(clean_df), figures_dir / "class_distribution.png")
    save_figure(correlation_figure, figures_dir / "correlations.png")
    save_figure(plot_directed_analysis(clean_df), figures_dir / "directed_analysis.png")
    save_figure(plot_confusion_matrices(artifacts), figures_dir / "confusion_matrices.png")
    save_figure(
        plot_feature_importance(logistic_coefficients, random_forest_importances),
        figures_dir / "feature_importance.png",
    )

    print("Assets generated successfully.")
    print(f"Processed dataset: {processed_data_dir / 'telecomx_clean.csv'}")
    print(f"Metrics report: {reports_dir / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
