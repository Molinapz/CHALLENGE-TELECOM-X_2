from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "TelecomX_2_ML.ipynb"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown_cell(
            """
            # Telecom X - Parte 2

            Este notebook reutiliza el flujo de limpieza y transformacion del proyecto anterior (`TELECOMX_LATAM-CHALLENGE`) para construir un pipeline predictivo de churn consistente y reproducible.

            ## Objetivos del reto

            - Usar el mismo dataset ya tratado en la Parte 1.
            - Preparar los datos para modelado eliminando columnas irrelevantes o redundantes.
            - Analizar correlaciones y patrones clave asociados al churn.
            - Entrenar al menos dos modelos de clasificacion.
            - Evaluar su desempeno y extraer conclusiones accionables para retencion.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            def find_repo_root(start: Path) -> Path:
                for candidate in [start, *start.parents]:
                    if (candidate / "src").exists() and (candidate / "data").exists():
                        return candidate
                raise FileNotFoundError("No se encontro la raiz del repositorio.")

            REPO_ROOT = find_repo_root(Path.cwd().resolve())
            SRC_DIR = REPO_ROOT / "src"
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))

            from telecomx_churn.pipeline import (
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
                MODEL_DROP_COLUMNS,
            )

            set_plot_theme()
            RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "TelecomX_Data.json"
            PROCESSED_DATA_PATH = REPO_ROOT / "data" / "processed" / "telecomx_clean.csv"
            FIGURES_DIR = REPO_ROOT / "reports" / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        markdown_cell(
            """
            ## 1. Extraccion del archivo tratado

            Se carga el JSON original copiado al nuevo repositorio y se aplica exactamente la misma limpieza y transformacion realizada en la Parte 1. Despues se persiste un CSV tratado para asegurar continuidad entre ambos retos.
            """
        ),
        code_cell(
            """
            clean_df = extract_and_clean_data(RAW_DATA_PATH)
            clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

            print(f"Dataset limpio guardado en: {PROCESSED_DATA_PATH}")
            print(f"Shape final: {clean_df.shape}")
            clean_df.head()
            """
        ),
        markdown_cell(
            """
            ## 2. Preparacion de datos

            Para el modelado se eliminan las siguientes columnas:

            - `customer_id`: identificador unico sin valor predictivo.
            - `churn`: version categorial del target ya convertida a `churn_flag`.
            - `senior_citizen_label`: duplicado de `senior_citizen`.
            - `daily_charges`: transformacion lineal exacta de `monthly_charges`, removida para reducir colinealidad.
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                {
                    "column_removed": MODEL_DROP_COLUMNS,
                    "reason": [
                        "Identificador unico",
                        "Target duplicado en texto",
                        "Duplicado categorico de una variable binaria",
                        "Variable redundante derivada de monthly_charges",
                    ],
                }
            )
            """
        ),
        code_cell(
            """
            preprocessing_summary = get_preprocessing_summary(clean_df)
            preprocessing_summary
            """
        ),
        markdown_cell(
            """
            La proporcion de churn es cercana al 26.5%, por lo que existe un desbalance moderado, pero no extremo. En lugar de aplicar SMOTE u oversampling, se utiliza:

            - separacion estratificada entre entrenamiento y prueba
            - `class_weight` en ambos modelos
            """
        ),
        code_cell(
            """
            fig = plot_class_distribution(clean_df)
            fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 3. Correlacion y seleccion de variables
            """
        ),
        code_cell(
            """
            corr_fig, target_correlations = plot_correlations(clean_df)
            corr_fig.savefig(FIGURES_DIR / "correlations.png", dpi=200, bbox_inches="tight")
            plt.show()

            target_correlations.sort_values()
            """
        ),
        markdown_cell(
            """
            Las relaciones mas claras con churn son:

            - `tenure_months` y `total_charges` con correlacion negativa.
            - `monthly_charges`, `paperless_billing` y `senior_citizen` con correlacion positiva.
            - La ausencia de `online_security` y `tech_support` tambien eleva el riesgo.
            """
        ),
        markdown_cell(
            """
            ## 4. Analisis dirigido

            Siguiendo el tablero, se revisan las relaciones:

            - tiempo de contrato x churn
            - gasto total x churn
            """
        ),
        code_cell(
            """
            directed_fig = plot_directed_analysis(clean_df)
            directed_fig.savefig(FIGURES_DIR / "directed_analysis.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        code_cell(
            """
            contract_rates = get_categorical_rate_table(clean_df, "contract")
            payment_rates = get_categorical_rate_table(clean_df, "payment_method")
            internet_rates = get_categorical_rate_table(clean_df, "internet_service")
            numeric_summary = get_numeric_summary_by_churn(clean_df)

            display(contract_rates)
            display(payment_rates)
            display(internet_rates)
            display(numeric_summary)
            """
        ),
        markdown_cell(
            """
            ## 5. Modelado predictivo

            Se comparan dos enfoques:

            - **Regresion Logistica**: requiere estandarizacion.
            - **Random Forest**: no requiere escalamiento y captura relaciones no lineales.
            """
        ),
        code_cell(
            """
            X, y = prepare_modeling_dataset(clean_df)
            X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

            print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
            """
        ),
        code_cell(
            """
            metrics_df, artifacts = compare_models(X_train, X_test, y_train, y_test)
            metrics_df
            """
        ),
        code_cell(
            """
            confusion_fig = plot_confusion_matrices(artifacts)
            confusion_fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 6. Importancia de variables
            """
        ),
        code_cell(
            """
            logistic_coefficients, random_forest_importances = get_feature_importances(artifacts, top_n=12)

            feature_fig = plot_feature_importance(logistic_coefficients, random_forest_importances)
            feature_fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
            plt.show()

            display(logistic_coefficients.to_frame("coefficient"))
            display(random_forest_importances.to_frame("importance"))
            """
        ),
        markdown_cell(
            """
            ## 7. Conclusiones

            ### Hallazgos tecnicos

            - **Random Forest** obtiene el mejor equilibrio global entre exactitud y F1 en prueba.
            - **Regresion Logistica** mantiene el mejor recall, por lo que resulta util si el negocio prioriza detectar la mayor cantidad posible de clientes con riesgo de cancelar.
            - El ajuste del bosque aleatorio con profundidad limitada reduce el sobreajuste observado en una version mas compleja.

            ### Variables mas influyentes

            - Contrato **Month-to-month**
            - Menor **antiguedad**
            - Mayor **cargo total / mensual**
            - Ausencia de **online security** y **tech support**
            - Servicio de internet **Fiber optic**
            - Metodo de pago **Electronic check**

            ### Recomendaciones de negocio

            1. Priorizar campanas de retencion temprana para clientes con menos antiguedad y contrato mensual.
            2. Ofrecer bundles con seguridad y soporte tecnico para clientes de Fiber optic.
            3. Revisar fricciones del canal `Electronic check`, que concentra la tasa mas alta de cancelacion.
            4. Usar la regresion logistica como modelo de alerta temprana y el Random Forest como referencia operativa por su mejor balance general.
            """
        ),
    ]

    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
        nbf.write(notebook, file)

    print(f"Notebook generated at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
