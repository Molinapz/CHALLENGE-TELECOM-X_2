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

ESTADO_ALEATORIO = 42
COLUMNA_OBJETIVO = "cancelacion_binaria"
COLUMNAS_DESCARTADAS_MODELO = ["id_cliente", "cancelacion", "etiqueta_adulto_mayor", "cargos_diarios"]


@dataclass
class ArtefactosModelo:
    nombre: str
    flujo_ajustado: Pipeline
    metricas: pd.DataFrame
    matrices_confusion: dict[str, np.ndarray]
    predicciones_entrenamiento: np.ndarray
    predicciones_prueba: np.ndarray


def definir_tema_graficos() -> None:
    sns.set_theme(style="whitegrid", palette="crest", font_scale=1.0)


def extraer_y_limpiar_datos(ruta_json_bruto: Path | str) -> pd.DataFrame:
    ruta_json_bruto = Path(ruta_json_bruto)
    datos_brutos = pd.read_json(ruta_json_bruto)

    datos_normalizados = pd.json_normalize(datos_brutos.to_dict(orient="records"))
    datos_normalizados.columns = (
        datos_normalizados.columns.str.replace("customer.", "", regex=False)
        .str.replace("phone.", "", regex=False)
        .str.replace("internet.", "", regex=False)
        .str.replace("account.", "", regex=False)
        .str.replace("Charges.", "Charges_", regex=False)
    )

    mapa_columnas = {
        "customerID": "id_cliente",
        "Churn": "cancelacion",
        "gender": "genero",
        "SeniorCitizen": "adulto_mayor",
        "Partner": "pareja",
        "Dependents": "dependientes",
        "tenure": "meses_antiguedad",
        "PhoneService": "servicio_telefonico",
        "MultipleLines": "multiples_lineas",
        "InternetService": "servicio_internet",
        "OnlineSecurity": "seguridad_en_linea",
        "OnlineBackup": "respaldo_en_linea",
        "DeviceProtection": "proteccion_dispositivo",
        "TechSupport": "soporte_tecnico",
        "StreamingTV": "tv_streaming",
        "StreamingMovies": "peliculas_streaming",
        "Contract": "tipo_contrato",
        "PaperlessBilling": "factura_digital",
        "PaymentMethod": "metodo_pago",
        "Charges_Monthly": "cargos_mensuales",
        "Charges_Total": "cargos_totales",
    }

    datos = datos_normalizados.rename(columns=mapa_columnas).copy()
    columnas_objeto = datos.select_dtypes(include=["object", "string"]).columns
    datos[columnas_objeto] = datos[columnas_objeto].apply(lambda columna: columna.str.strip())

    datos["cancelacion"] = datos["cancelacion"].replace("", np.nan)
    datos["cargos_totales"] = pd.to_numeric(datos["cargos_totales"], errors="coerce")

    mascara_cero_antiguedad = datos["cargos_totales"].isna() & datos["meses_antiguedad"].eq(0)
    datos.loc[mascara_cero_antiguedad, "cargos_totales"] = 0

    datos = datos.dropna(subset=["cancelacion"]).copy()
    datos["adulto_mayor"] = datos["adulto_mayor"].astype(int)
    mapas_valores = {
        "cancelacion": {"Yes": "Si", "No": "No"},
        "genero": {"Male": "Masculino", "Female": "Femenino"},
        "pareja": {"Yes": "Si", "No": "No"},
        "dependientes": {"Yes": "Si", "No": "No"},
        "servicio_telefonico": {"Yes": "Si", "No": "No"},
        "multiples_lineas": {
            "Yes": "Si",
            "No": "No",
            "No phone service": "Sin servicio telefonico",
        },
        "servicio_internet": {
            "Fiber optic": "Fibra optica",
            "DSL": "DSL",
            "No": "Sin servicio",
        },
        "seguridad_en_linea": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "respaldo_en_linea": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "proteccion_dispositivo": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "soporte_tecnico": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "tv_streaming": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "peliculas_streaming": {
            "Yes": "Si",
            "No": "No",
            "No internet service": "Sin servicio de internet",
        },
        "tipo_contrato": {
            "Month-to-month": "Mes a mes",
            "One year": "Un ano",
            "Two year": "Dos anos",
        },
        "factura_digital": {"Yes": "Si", "No": "No"},
        "metodo_pago": {
            "Electronic check": "Cheque electronico",
            "Mailed check": "Cheque enviado por correo",
            "Bank transfer (automatic)": "Transferencia bancaria automatica",
            "Credit card (automatic)": "Tarjeta de credito automatica",
        },
    }

    for columna, mapa in mapas_valores.items():
        datos[columna] = datos[columna].replace(mapa)

    datos["cancelacion_binaria"] = datos["cancelacion"].map({"Si": 1, "No": 0})
    datos["cargos_diarios"] = datos["cargos_mensuales"] / 30

    columnas_servicios = [
        "servicio_telefonico",
        "multiples_lineas",
        "seguridad_en_linea",
        "respaldo_en_linea",
        "proteccion_dispositivo",
        "soporte_tecnico",
        "tv_streaming",
        "peliculas_streaming",
    ]

    datos["total_servicios"] = (
        datos[columnas_servicios].eq("Si").sum(axis=1)
        + datos["servicio_internet"].ne("Sin servicio").astype(int)
    )
    datos["etiqueta_adulto_mayor"] = datos["adulto_mayor"].map({0: "No", 1: "Si"})

    return datos


def obtener_resumen_preparacion(datos_limpios: pd.DataFrame) -> pd.DataFrame:
    conteos = datos_limpios[COLUMNA_OBJETIVO].value_counts().sort_index()
    resumen = (
        datos_limpios[COLUMNA_OBJETIVO]
        .value_counts(normalize=True)
        .sort_index()
        .rename_axis(COLUMNA_OBJETIVO)
        .reset_index(name="proporcion")
    )
    resumen["clase"] = resumen[COLUMNA_OBJETIVO].map({0: "No cancela", 1: "Cancela"})
    resumen["clientes"] = resumen[COLUMNA_OBJETIVO].map(conteos).astype(int)
    resumen["proporcion_pct"] = (resumen["proporcion"] * 100).round(2)
    return resumen[["clase", "clientes", "proporcion_pct"]]


def construir_marco_correlacion_objetivo(datos_limpios: pd.DataFrame) -> pd.DataFrame:
    marco_correlacion = datos_limpios[
        [
            COLUMNA_OBJETIVO,
            "adulto_mayor",
            "meses_antiguedad",
            "cargos_mensuales",
            "cargos_totales",
            "cargos_diarios",
            "total_servicios",
        ]
    ].copy()

    variables_binarias = [
        "pareja",
        "dependientes",
        "factura_digital",
        "servicio_telefonico",
        "multiples_lineas",
        "seguridad_en_linea",
        "respaldo_en_linea",
        "proteccion_dispositivo",
        "soporte_tecnico",
        "tv_streaming",
        "peliculas_streaming",
    ]

    for columna in variables_binarias:
        marco_correlacion[columna] = datos_limpios[columna].map(
            {
                "Si": 1,
                "No": 0,
                "Sin servicio de internet": 0,
                "Sin servicio telefonico": 0,
            }
        )

    return marco_correlacion


def obtener_tabla_tasa_categorica(datos_limpios: pd.DataFrame, columna: str) -> pd.DataFrame:
    resumen = (
        datos_limpios.groupby(columna)
        .agg(clientes=(COLUMNA_OBJETIVO, "size"), tasa_cancelacion=(COLUMNA_OBJETIVO, "mean"))
        .sort_values("tasa_cancelacion", ascending=False)
        .reset_index()
    )
    resumen["tasa_cancelacion"] = (resumen["tasa_cancelacion"] * 100).round(2)
    return resumen


def obtener_resumen_numerico_por_cancelacion(datos_limpios: pd.DataFrame) -> pd.DataFrame:
    columnas_numericas = ["meses_antiguedad", "cargos_mensuales", "cargos_totales", "total_servicios"]
    resumen = (
        datos_limpios.groupby("cancelacion")[columnas_numericas]
        .agg(["mean", "median"])
        .round(2)
    )
    resumen.columns = ["_".join(columna).strip() for columna in resumen.columns.to_flat_index()]
    return resumen.reset_index()


def preparar_datos_modelado(datos_limpios: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    datos_modelado = datos_limpios.drop(columns=COLUMNAS_DESCARTADAS_MODELO)
    X = datos_modelado.drop(columns=[COLUMNA_OBJETIVO])
    y = datos_modelado[COLUMNA_OBJETIVO]
    return X, y

def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
    proporcion_prueba: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=proporcion_prueba,
        random_state=ESTADO_ALEATORIO,
        stratify=y,
    )


def _construir_preprocesadores(X: pd.DataFrame) -> tuple[list[str], list[str], ColumnTransformer, ColumnTransformer]:
    variables_numericas = X.select_dtypes(include="number").columns.tolist()
    variables_categoricas = X.select_dtypes(exclude="number").columns.tolist()

    preprocesador_escalado = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="median")),
                        ("escalador", StandardScaler()),
                    ]
                ),
                variables_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="most_frequent")),
                        ("codificador", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                variables_categoricas,
            ),
        ],
        verbose_feature_names_out=False,
    )

    preprocesador_arbol = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(steps=[("imputador", SimpleImputer(strategy="median"))]),
                variables_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="most_frequent")),
                        ("codificador", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                variables_categoricas,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return variables_numericas, variables_categoricas, preprocesador_escalado, preprocesador_arbol


def construir_modelos(X: pd.DataFrame) -> dict[str, Pipeline]:
    _, _, preprocesador_escalado, preprocesador_arbol = _construir_preprocesadores(X)

    return {
        "Regresion Logistica": Pipeline(
            steps=[
                ("preparacion", preprocesador_escalado),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=ESTADO_ALEATORIO,
                    ),
                ),
            ]
        ),
        "Bosque Aleatorio": Pipeline(
            steps=[
                ("preparacion", preprocesador_arbol),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=4,
                        min_samples_split=12,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=ESTADO_ALEATORIO,
                    ),
                ),
            ]
        ),
    }


def comparar_modelos(
    X_entrenamiento: pd.DataFrame,
    X_prueba: pd.DataFrame,
    y_entrenamiento: pd.Series,
    y_prueba: pd.Series,
) -> tuple[pd.DataFrame, dict[str, ArtefactosModelo]]:
    filas_metricas: list[dict[str, Any]] = []
    artefactos: dict[str, ArtefactosModelo] = {}

    for nombre_modelo, flujo in construir_modelos(X_entrenamiento).items():
        flujo.fit(X_entrenamiento, y_entrenamiento)
        predicciones = {
            "entrenamiento": flujo.predict(X_entrenamiento),
            "prueba": flujo.predict(X_prueba),
        }
        metricas_division: list[dict[str, Any]] = []
        matrices_confusion: dict[str, np.ndarray] = {}

        for nombre_division, y_real, y_predicho in (
            ("entrenamiento", y_entrenamiento, predicciones["entrenamiento"]),
            ("prueba", y_prueba, predicciones["prueba"]),
        ):
            vn, fp, fn, vp = confusion_matrix(y_real, y_predicho).ravel()
            fila = {
                "modelo": nombre_modelo,
                "division": nombre_division,
                "exactitud": round(accuracy_score(y_real, y_predicho), 4),
                "precision": round(precision_score(y_real, y_predicho), 4),
                "recall": round(recall_score(y_real, y_predicho), 4),
                "puntaje_f1": round(f1_score(y_real, y_predicho), 4),
                "verdaderos_negativos": int(vn),
                "falsos_positivos": int(fp),
                "falsos_negativos": int(fn),
                "verdaderos_positivos": int(vp),
            }
            filas_metricas.append(fila)
            metricas_division.append(fila)
            matrices_confusion[nombre_division] = np.array([[vn, fp], [fn, vp]])

        artefactos[nombre_modelo] = ArtefactosModelo(
            nombre=nombre_modelo,
            flujo_ajustado=flujo,
            metricas=pd.DataFrame(metricas_division),
            matrices_confusion=matrices_confusion,
            predicciones_entrenamiento=predicciones["entrenamiento"],
            predicciones_prueba=predicciones["prueba"],
        )

    comparacion = pd.DataFrame(filas_metricas).sort_values(["division", "puntaje_f1"], ascending=[True, False])
    return comparacion, artefactos


def obtener_importancia_variables(
    artefactos: dict[str, ArtefactosModelo],
    top_n: int = 12,
) -> tuple[pd.Series, pd.Series]:
    artefacto_logit = artefactos["Regresion Logistica"].flujo_ajustado
    variables_logit = artefacto_logit.named_steps["preparacion"].get_feature_names_out()
    coeficientes_logit = (
        pd.Series(artefacto_logit.named_steps["modelo"].coef_[0], index=variables_logit)
        .sort_values(key=np.abs, ascending=False)
        .head(top_n)
    )

    artefacto_bosque = artefactos["Bosque Aleatorio"].flujo_ajustado
    variables_bosque = artefacto_bosque.named_steps["preparacion"].get_feature_names_out()
    importancias_bosque = (
        pd.Series(
            artefacto_bosque.named_steps["modelo"].feature_importances_,
            index=variables_bosque,
        )
        .sort_values(ascending=False)
        .head(top_n)
    )

    return coeficientes_logit, importancias_bosque

def graficar_distribucion_clases(datos_limpios: pd.DataFrame) -> plt.Figure:
    distribucion = (
        datos_limpios["cancelacion"]
        .value_counts()
        .rename_axis("cancelacion")
        .reset_index(name="clientes")
    )
    distribucion["estado_cancelacion"] = distribucion["cancelacion"].map({"Si": "Cancela", "No": "No cancela"})

    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=distribucion,
        x="estado_cancelacion",
        y="clientes",
        hue="estado_cancelacion",
        dodge=False,
        legend=False,
        ax=ejes[0],
        palette=["#264653", "#E76F51"],
    )
    ejes[0].set_title("Distribucion de cancelacion")
    ejes[0].set_xlabel("")
    ejes[0].set_ylabel("Clientes")

    ejes[1].pie(
        distribucion["clientes"],
        labels=distribucion["estado_cancelacion"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#264653", "#E76F51"],
    )
    ejes[1].set_title("Proporcion de cancelacion")

    figura.tight_layout()
    return figura


def graficar_correlaciones(datos_limpios: pd.DataFrame) -> tuple[plt.Figure, pd.Series]:
    marco_correlacion = construir_marco_correlacion_objetivo(datos_limpios)
    correlaciones_objetivo = marco_correlacion.corr(numeric_only=True)[COLUMNA_OBJETIVO].sort_values()

    figura, ejes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(marco_correlacion.corr(numeric_only=True), cmap="vlag", center=0, ax=ejes[0])
    ejes[0].set_title("Matriz de correlacion")

    correlaciones_ordenadas = correlaciones_objetivo.drop(COLUMNA_OBJETIVO)
    colores = ["#1B4332" if valor < 0 else "#D62828" for valor in correlaciones_ordenadas.values]
    ejes[1].barh(correlaciones_ordenadas.index, correlaciones_ordenadas.values, color=colores)
    ejes[1].set_title("Correlacion con cancelacion_binaria")
    ejes[1].set_xlabel("Correlacion")
    ejes[1].set_ylabel("")
    ejes[1].axvline(0, color="#495057", linewidth=1)

    figura.tight_layout()
    return figura, correlaciones_objetivo


def graficar_analisis_dirigido(datos_limpios: pd.DataFrame) -> plt.Figure:
    resumen_contrato = obtener_tabla_tasa_categorica(datos_limpios, "tipo_contrato")

    figura, ejes = plt.subplots(2, 2, figsize=(16, 10))
    sns.boxplot(
        data=datos_limpios,
        x="cancelacion",
        y="meses_antiguedad",
        hue="cancelacion",
        dodge=False,
        legend=False,
        ax=ejes[0, 0],
        palette=["#2A9D8F", "#E76F51"],
    )
    ejes[0, 0].set_title("Antiguedad del cliente x cancelacion")
    ejes[0, 0].set_xlabel("")
    ejes[0, 0].set_ylabel("Meses")

    sns.boxplot(
        data=datos_limpios,
        x="cancelacion",
        y="cargos_totales",
        hue="cancelacion",
        dodge=False,
        legend=False,
        ax=ejes[0, 1],
        palette=["#2A9D8F", "#E76F51"],
    )
    ejes[0, 1].set_title("Cargo total x cancelacion")
    ejes[0, 1].set_xlabel("")
    ejes[0, 1].set_ylabel("Cargo total")

    sns.violinplot(
        data=datos_limpios,
        x="cancelacion",
        y="cargos_mensuales",
        hue="cancelacion",
        dodge=False,
        legend=False,
        ax=ejes[1, 0],
        palette=["#2A9D8F", "#E76F51"],
    )
    ejes[1, 0].set_title("Cargo mensual x cancelacion")
    ejes[1, 0].set_xlabel("")
    ejes[1, 0].set_ylabel("Cargo mensual")

    sns.barplot(data=resumen_contrato, x="tipo_contrato", y="tasa_cancelacion", ax=ejes[1, 1], color="#2A9D8F")
    ejes[1, 1].set_title("Tasa de cancelacion por tipo de contrato")
    ejes[1, 1].set_xlabel("")
    ejes[1, 1].set_ylabel("Cancelacion (%)")
    ejes[1, 1].tick_params(axis="x", rotation=15)

    figura.tight_layout()
    return figura

def graficar_matrices_confusion(artefactos: dict[str, ArtefactosModelo]) -> plt.Figure:
    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))

    for eje, nombre_modelo in zip(ejes, ("Regresion Logistica", "Bosque Aleatorio")):
        matriz = artefactos[nombre_modelo].matrices_confusion["prueba"]
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=eje)
        eje.set_title(f"Matriz de confusion - {nombre_modelo}")
        eje.set_xlabel("Prediccion")
        eje.set_ylabel("Real")
        eje.set_xticklabels(["No cancela", "Cancela"])
        eje.set_yticklabels(["No cancela", "Cancela"], rotation=0)

    figura.tight_layout()
    return figura


def graficar_importancia_variables(
    coeficientes_logit: pd.Series,
    importancias_bosque: pd.Series,
) -> plt.Figure:
    figura, ejes = plt.subplots(1, 2, figsize=(18, 7))

    marco_logit = coeficientes_logit.sort_values()
    colores_logit = ["#1D3557" if valor < 0 else "#E63946" for valor in marco_logit.values]
    ejes[0].barh(marco_logit.index, marco_logit.values, color=colores_logit)
    ejes[0].set_title("Coeficientes mas influyentes - Regresion Logistica")
    ejes[0].set_xlabel("Coeficiente")
    ejes[0].set_ylabel("")
    ejes[0].axvline(0, color="#495057", linewidth=1)

    marco_bosque = importancias_bosque.sort_values()
    ejes[1].barh(marco_bosque.index, marco_bosque.values, color="#2A9D8F")
    ejes[1].set_title("Importancia de variables - Bosque Aleatorio")
    ejes[1].set_xlabel("Importancia")
    ejes[1].set_ylabel("")

    figura.tight_layout()
    return figura
