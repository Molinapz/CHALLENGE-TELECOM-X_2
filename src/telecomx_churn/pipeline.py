from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = "churn_flag"
MODEL_DROP_COLUMNS = ["customer_id", "churn", "senior_citizen_label", "daily_charges"]


@dataclass
class ModelArtifacts:
    name: str
    fitted_pipeline: Pipeline
    metrics: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    train_predictions: np.ndarray
    test_predictions: np.ndarray


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.0)


def extract_and_clean_data(raw_json_path: Path | str) -> pd.DataFrame:
    raw_json_path = Path(raw_json_path)
    telecom_raw = pd.read_json(raw_json_path)

    telecom_df = pd.json_normalize(telecom_raw.to_dict(orient="records"))
    telecom_df.columns = (
        telecom_df.columns.str.replace("customer.", "", regex=False)
        .str.replace("phone.", "", regex=False)
        .str.replace("internet.", "", regex=False)
        .str.replace("account.", "", regex=False)
        .str.replace("Charges.", "Charges_", regex=False)
    )

    rename_map = {
        "customerID": "customer_id",
        "Churn": "churn",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure_months",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "Charges_Monthly": "monthly_charges",
        "Charges_Total": "total_charges",
    }

    df = telecom_df.rename(columns=rename_map).copy()
    object_columns = df.select_dtypes(include=["object", "string"]).columns
    df[object_columns] = df[object_columns].apply(lambda col: col.str.strip())

    df["churn"] = df["churn"].replace("", np.nan)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    mask_zero_tenure = df["total_charges"].isna() & df["tenure_months"].eq(0)
    df.loc[mask_zero_tenure, "total_charges"] = 0

    df = df.dropna(subset=["churn"]).copy()
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df["churn_flag"] = df["churn"].map({"Yes": 1, "No": 0})
    df["daily_charges"] = df["monthly_charges"] / 30

    service_columns = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    df["total_services"] = (
        df[service_columns].eq("Yes").sum(axis=1)
        + df["internet_service"].ne("No").astype(int)
    )
    df["senior_citizen_label"] = df["senior_citizen"].map({0: "No", 1: "Yes"})

    return df


def get_preprocessing_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    counts = clean_df[TARGET_COLUMN].value_counts().sort_index()
    class_share = (
        clean_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .rename_axis("churn_flag")
        .reset_index(name="share")
    )
    class_share["class_name"] = class_share["churn_flag"].map({0: "No", 1: "Yes"})
    class_share["clients"] = class_share["churn_flag"].map(counts).astype(int)
    class_share["share_pct"] = (class_share["share"] * 100).round(2)
    return class_share[["class_name", "clients", "share_pct"]]


def build_target_correlation_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    corr_df = clean_df[
        [
            TARGET_COLUMN,
            "senior_citizen",
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "daily_charges",
            "total_services",
        ]
    ].copy()

    binary_features = [
        "partner",
        "dependents",
        "paperless_billing",
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    for column in binary_features:
        corr_df[column] = clean_df[column].map(
            {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        )

    return corr_df


def get_categorical_rate_table(clean_df: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        clean_df.groupby(column)
        .agg(clients=(TARGET_COLUMN, "size"), churn_rate=(TARGET_COLUMN, "mean"))
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(2)
    return summary


def get_numeric_summary_by_churn(clean_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["tenure_months", "monthly_charges", "total_charges", "total_services"]
    summary = (
        clean_df.groupby("churn")[numeric_columns]
        .agg(["mean", "median"])
        .round(2)
    )
    summary.columns = ["_".join(column).strip() for column in summary.columns.to_flat_index()]
    return summary.reset_index()


def prepare_modeling_dataset(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    modeling_df = clean_df.drop(columns=MODEL_DROP_COLUMNS)
    X = modeling_df.drop(columns=[TARGET_COLUMN])
    y = modeling_df[TARGET_COLUMN]
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _build_preprocessors(X: pd.DataFrame) -> tuple[list[str], list[str], ColumnTransformer, ColumnTransformer]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return numeric_features, categorical_features, scaled_preprocessor, tree_preprocessor


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    _, _, scaled_preprocessor, tree_preprocessor = _build_preprocessors(X)

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=4,
                        min_samples_split=12,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, ModelArtifacts]]:
    metrics_rows: list[dict[str, Any]] = []
    artifacts: dict[str, ModelArtifacts] = {}

    for model_name, pipeline in build_models(X_train).items():
        pipeline.fit(X_train, y_train)
        split_predictions = {
            "train": pipeline.predict(X_train),
            "test": pipeline.predict(X_test),
        }
        split_metrics: list[dict[str, Any]] = []
        confusion_matrices: dict[str, np.ndarray] = {}

        for split_name, y_true, y_pred in (
            ("train", y_train, split_predictions["train"]),
            ("test", y_test, split_predictions["test"]),
        ):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics_row = {
                "model": model_name,
                "split": split_name,
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred), 4),
                "recall": round(recall_score(y_true, y_pred), 4),
                "f1_score": round(f1_score(y_true, y_pred), 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            metrics_rows.append(metrics_row)
            split_metrics.append(metrics_row)
            confusion_matrices[split_name] = np.array([[tn, fp], [fn, tp]])

        artifacts[model_name] = ModelArtifacts(
            name=model_name,
            fitted_pipeline=pipeline,
            metrics=pd.DataFrame(split_metrics),
            confusion_matrices=confusion_matrices,
            train_predictions=split_predictions["train"],
            test_predictions=split_predictions["test"],
        )

    comparison_df = pd.DataFrame(metrics_rows).sort_values(["split", "f1_score"], ascending=[True, False])
    return comparison_df, artifacts


def get_feature_importances(
    artifacts: dict[str, ModelArtifacts],
    top_n: int = 12,
) -> tuple[pd.Series, pd.Series]:
    logistic_artifact = artifacts["Logistic Regression"].fitted_pipeline
    logistic_features = logistic_artifact.named_steps["prep"].get_feature_names_out()
    logistic_coefficients = (
        pd.Series(logistic_artifact.named_steps["model"].coef_[0], index=logistic_features)
        .sort_values(key=np.abs, ascending=False)
        .head(top_n)
    )

    random_forest_artifact = artifacts["Random Forest"].fitted_pipeline
    random_forest_features = random_forest_artifact.named_steps["prep"].get_feature_names_out()
    random_forest_importances = (
        pd.Series(
            random_forest_artifact.named_steps["model"].feature_importances_,
            index=random_forest_features,
        )
        .sort_values(ascending=False)
        .head(top_n)
    )

    return logistic_coefficients, random_forest_importances


def plot_class_distribution(clean_df: pd.DataFrame) -> plt.Figure:
    distribution = (
        clean_df["churn"]
        .value_counts()
        .rename_axis("churn")
        .reset_index(name="clients")
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=distribution, x="churn", y="clients", hue="churn", dodge=False, legend=False, ax=axes[0], palette=["#264653", "#E76F51"])
    axes[0].set_title("Distribucion de churn")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Clientes")

    axes[1].pie(
        distribution["clients"],
        labels=distribution["churn"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#264653", "#E76F51"],
    )
    axes[1].set_title("Proporcion de churn")

    figure.tight_layout()
    return figure


def plot_correlations(clean_df: pd.DataFrame) -> tuple[plt.Figure, pd.Series]:
    corr_df = build_target_correlation_frame(clean_df)
    target_correlations = corr_df.corr(numeric_only=True)[TARGET_COLUMN].sort_values()

    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(corr_df.corr(numeric_only=True), cmap="vlag", center=0, ax=axes[0])
    axes[0].set_title("Matriz de correlacion")

    sorted_correlations = target_correlations.drop(TARGET_COLUMN)
    colors = ["#1B4332" if value < 0 else "#D62828" for value in sorted_correlations.values]
    axes[1].barh(sorted_correlations.index, sorted_correlations.values, color=colors)
    axes[1].set_title("Correlacion con churn_flag")
    axes[1].set_xlabel("Correlacion")
    axes[1].set_ylabel("")
    axes[1].axvline(0, color="#495057", linewidth=1)

    figure.tight_layout()
    return figure, target_correlations


def plot_directed_analysis(clean_df: pd.DataFrame) -> plt.Figure:
    contract_summary = get_categorical_rate_table(clean_df, "contract")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.boxplot(data=clean_df, x="churn", y="tenure_months", hue="churn", dodge=False, legend=False, ax=axes[0, 0], palette=["#2A9D8F", "#E76F51"])
    axes[0, 0].set_title("Antiguedad del cliente x churn")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Meses")

    sns.boxplot(data=clean_df, x="churn", y="total_charges", hue="churn", dodge=False, legend=False, ax=axes[0, 1], palette=["#2A9D8F", "#E76F51"])
    axes[0, 1].set_title("Cargo total x churn")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Cargo total")

    sns.violinplot(data=clean_df, x="churn", y="monthly_charges", hue="churn", dodge=False, legend=False, ax=axes[1, 0], palette=["#2A9D8F", "#E76F51"])
    axes[1, 0].set_title("Cargo mensual x churn")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Cargo mensual")

    sns.barplot(data=contract_summary, x="contract", y="churn_rate", ax=axes[1, 1], color="#2A9D8F")
    axes[1, 1].set_title("Tasa de churn por contrato")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Churn (%)")
    axes[1, 1].tick_params(axis="x", rotation=15)

    figure.tight_layout()
    return figure


def plot_confusion_matrices(artifacts: dict[str, ModelArtifacts]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    for axis, model_name in zip(axes, ("Logistic Regression", "Random Forest")):
        matrix = artifacts[model_name].confusion_matrices["test"]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"Matriz de confusion - {model_name}")
        axis.set_xlabel("Prediccion")
        axis.set_ylabel("Real")
        axis.set_xticklabels(["No churn", "Churn"])
        axis.set_yticklabels(["No churn", "Churn"], rotation=0)

    figure.tight_layout()
    return figure


def plot_feature_importance(
    logistic_coefficients: pd.Series,
    random_forest_importances: pd.Series,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))

    logistic_frame = logistic_coefficients.sort_values()
    logistic_colors = ["#1D3557" if value < 0 else "#E63946" for value in logistic_frame.values]
    axes[0].barh(logistic_frame.index, logistic_frame.values, color=logistic_colors)
    axes[0].set_title("Coeficientes mas influyentes - Regresion logistica")
    axes[0].set_xlabel("Coeficiente")
    axes[0].set_ylabel("")
    axes[0].axvline(0, color="#495057", linewidth=1)

    rf_frame = random_forest_importances.sort_values()
    axes[1].barh(rf_frame.index, rf_frame.values, color="#2A9D8F")
    axes[1].set_title("Importancia de variables - Random Forest")
    axes[1].set_xlabel("Importancia")
    axes[1].set_ylabel("")

    figure.tight_layout()
    return figure
