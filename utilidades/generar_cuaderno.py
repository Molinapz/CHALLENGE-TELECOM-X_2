from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
RUTA_CUADERNO = RAIZ_REPOSITORIO / "cuadernos" / "TelecomX_2_Cancelacion.ipynb"


def celda_markdown(texto: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(texto).strip())


def celda_codigo(texto: str):
    return nbf.v4.new_code_cell(textwrap.dedent(texto).strip())


def principal() -> None:
    RUTA_CUADERNO.parent.mkdir(parents=True, exist_ok=True)

    cuaderno = nbf.v4.new_notebook()
    cuaderno["cells"] = [
        celda_markdown(
            """
            # Telecom X - Parte 2

            Este cuaderno reutiliza exactamente la limpieza y transformacion del proyecto anterior para mantener la continuidad del analisis y construir un flujo predictivo de cancelacion consistente.

            ## Objetivos del reto

            - Usar el mismo conjunto de datos tratado en la Parte 1.
            - Preparar los datos para el modelado eliminando variables irrelevantes o redundantes.
            - Analizar correlaciones y patrones asociados a la cancelacion.
            - Entrenar al menos dos modelos de clasificacion.
            - Evaluar su desempeno e interpretar las variables mas influyentes.
            """
        ),
        celda_codigo(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            def encontrar_raiz_repositorio(inicio: Path) -> Path:
                for candidata in [inicio, *inicio.parents]:
                    if (candidata / "codigo").exists() and (candidata / "datos").exists():
                        return candidata
                raise FileNotFoundError("No se encontro la raiz del repositorio.")

            RAIZ_REPOSITORIO = encontrar_raiz_repositorio(Path.cwd().resolve())
            RUTA_CODIGO = RAIZ_REPOSITORIO / "codigo"
            if str(RUTA_CODIGO) not in sys.path:
                sys.path.insert(0, str(RUTA_CODIGO))

            from telecomx_cancelacion import (
                COLUMNAS_DESCARTADAS_MODELO,
                comparar_modelos,
                definir_tema_graficos,
                dividir_datos,
                extraer_y_limpiar_datos,
                graficar_analisis_dirigido,
                graficar_correlaciones,
                graficar_distribucion_clases,
                graficar_importancia_variables,
                graficar_matrices_confusion,
                obtener_importancia_variables,
                obtener_resumen_numerico_por_cancelacion,
                obtener_resumen_preparacion,
                obtener_tabla_tasa_categorica,
                preparar_datos_modelado,
            )

            definir_tema_graficos()
            RUTA_JSON_BRUTO = RAIZ_REPOSITORIO / "datos" / "brutos" / "TelecomX_Datos.json"
            RUTA_CSV_LIMPIO = RAIZ_REPOSITORIO / "datos" / "procesados" / "telecomx_limpio.csv"
            RUTA_GRAFICOS = RAIZ_REPOSITORIO / "informes" / "graficos"
            RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
            """
        ),
        celda_markdown(
            """
            ## 1. Extraccion del archivo tratado

            Se toma el JSON base y se vuelve a aplicar el mismo flujo de limpieza de la Parte 1 para garantizar continuidad. Despues se persiste un CSV limpio con nombres en espanol.
            """
        ),
        celda_codigo(
            """
            datos_limpios = extraer_y_limpiar_datos(RUTA_JSON_BRUTO)
            datos_limpios.to_csv(RUTA_CSV_LIMPIO, index=False)

            print(f"Datos limpios guardados en: {RUTA_CSV_LIMPIO}")
            print(f"Dimensiones finales: {datos_limpios.shape}")
            datos_limpios.head()
            """
        ),
        celda_markdown(
            """
            ## 2. Preparacion de datos

            Para modelar se descartan las siguientes columnas:

            - `id_cliente`: identificador unico sin valor predictivo.
            - `cancelacion`: version textual de la variable objetivo.
            - `etiqueta_adulto_mayor`: duplicado categorico de `adulto_mayor`.
            - `cargos_diarios`: derivada exacta de `cargos_mensuales`, retirada para evitar redundancia.
            """
        ),
        celda_codigo(
            """
            pd.DataFrame(
                {
                    "columna_descartada": COLUMNAS_DESCARTADAS_MODELO,
                    "motivo": [
                        "Identificador unico",
                        "Objetivo duplicado en texto",
                        "Duplicado categorico de una variable binaria",
                        "Variable redundante derivada de cargos_mensuales",
                    ],
                }
            )
            """
        ),
        celda_codigo(
            """
            resumen_preparacion = obtener_resumen_preparacion(datos_limpios)
            resumen_preparacion
            """
        ),
        celda_markdown(
            """
            La distribucion de cancelacion es moderadamente desbalanceada, pero no extrema. En lugar de tecnicas de sobremuestreo, se utiliza:

            - division estratificada entre entrenamiento y prueba
            - `class_weight` en ambos modelos
            """
        ),
        celda_codigo(
            """
            figura_distribucion = graficar_distribucion_clases(datos_limpios)
            figura_distribucion.savefig(RUTA_GRAFICOS / "distribucion_clases.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        celda_markdown(
            """
            ## 3. Correlacion y seleccion de variables
            """
        ),
        celda_codigo(
            """
            figura_correlaciones, correlaciones_objetivo = graficar_correlaciones(datos_limpios)
            figura_correlaciones.savefig(RUTA_GRAFICOS / "correlaciones_cancelacion.png", dpi=200, bbox_inches="tight")
            plt.show()

            correlaciones_objetivo.sort_values()
            """
        ),
        celda_markdown(
            """
            Las relaciones mas claras con la cancelacion son:

            - `meses_antiguedad` y `cargos_totales` con correlacion negativa.
            - `cargos_mensuales`, `factura_digital` y `adulto_mayor` con correlacion positiva.
            - La ausencia de `seguridad_en_linea` y `soporte_tecnico` eleva el riesgo.
            """
        ),
        celda_markdown(
            """
            ## 4. Analisis dirigido

            Siguiendo el tablero, se revisan las relaciones entre permanencia, gasto y cancelacion.
            """
        ),
        celda_codigo(
            """
            figura_analisis = graficar_analisis_dirigido(datos_limpios)
            figura_analisis.savefig(RUTA_GRAFICOS / "analisis_dirigido.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        celda_codigo(
            """
            tasas_contrato = obtener_tabla_tasa_categorica(datos_limpios, "tipo_contrato")
            tasas_pago = obtener_tabla_tasa_categorica(datos_limpios, "metodo_pago")
            tasas_internet = obtener_tabla_tasa_categorica(datos_limpios, "servicio_internet")
            resumen_numerico = obtener_resumen_numerico_por_cancelacion(datos_limpios)

            display(tasas_contrato)
            display(tasas_pago)
            display(tasas_internet)
            display(resumen_numerico)
            """
        ),
        celda_markdown(
            """
            ## 5. Modelado predictivo

            Se comparan dos enfoques:

            - **Regresion Logistica**: sensible a la escala, por eso usa estandarizacion.
            - **Bosque Aleatorio**: no necesita escalamiento y captura relaciones no lineales.
            """
        ),
        celda_codigo(
            """
            X, y = preparar_datos_modelado(datos_limpios)
            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(X, y, proporcion_prueba=0.2)

            print(f"Entrenamiento: {X_entrenamiento.shape}, Prueba: {X_prueba.shape}")
            """
        ),
        celda_codigo(
            """
            metricas_modelos, artefactos = comparar_modelos(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
            metricas_modelos
            """
        ),
        celda_codigo(
            """
            figura_confusion = graficar_matrices_confusion(artefactos)
            figura_confusion.savefig(RUTA_GRAFICOS / "matrices_confusion.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        celda_markdown(
            """
            ## 6. Importancia de variables
            """
        ),
        celda_codigo(
            """
            coeficientes_logit, importancias_bosque = obtener_importancia_variables(artefactos, top_n=12)

            figura_importancia = graficar_importancia_variables(coeficientes_logit, importancias_bosque)
            figura_importancia.savefig(RUTA_GRAFICOS / "importancia_variables.png", dpi=200, bbox_inches="tight")
            plt.show()

            display(coeficientes_logit.to_frame("coeficiente"))
            display(importancias_bosque.to_frame("importancia"))
            """
        ),
        celda_markdown(
            """
            ## 7. Conclusiones

            ### Hallazgos tecnicos

            - **Bosque Aleatorio** obtuvo el mejor equilibrio entre exactitud y puntaje F1 en prueba.
            - **Regresion Logistica** mantuvo el mejor recall, util cuando el negocio prioriza detectar la mayor cantidad posible de clientes en riesgo.
            - Limitar la complejidad del bosque redujo el sobreajuste frente a una configuracion mas agresiva.

            ### Variables mas influyentes

            - Tipo de contrato **Mes a mes**
            - Menor **meses_antiguedad**
            - Mayor nivel de **cargos**
            - Ausencia de **seguridad_en_linea** y **soporte_tecnico**
            - Servicio **Fibra optica**
            - Metodo de pago **Cheque electronico**

            ### Recomendaciones de negocio

            1. Priorizar campanas tempranas para clientes nuevos con contrato mes a mes.
            2. Ofrecer paquetes con seguridad y soporte para clientes de fibra optica.
            3. Revisar fricciones del canal `Cheque electronico`, que concentra la tasa mas alta de cancelacion.
            4. Usar la Regresion Logistica como alerta temprana y el Bosque Aleatorio como referencia operativa por su mejor balance global.
            """
        ),
    ]

    cuaderno["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cuaderno["metadata"]["language_info"] = {"name": "python", "version": "3.11"}

    with RUTA_CUADERNO.open("w", encoding="utf-8") as archivo:
        nbf.write(cuaderno, archivo)

    print(f"Cuaderno generado en: {RUTA_CUADERNO}")


if __name__ == "__main__":
    principal()
