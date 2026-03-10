from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
RUTA_CODIGO = RAIZ_REPOSITORIO / "codigo"
if str(RUTA_CODIGO) not in sys.path:
    sys.path.insert(0, str(RUTA_CODIGO))

from telecomx_cancelacion.flujo import (  # noqa: E402
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


def guardar_figura(figura: plt.Figure, ruta: Path) -> None:
    figura.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(figura)


def principal() -> None:
    definir_tema_graficos()

    ruta_fuente_json = RAIZ_REPOSITORIO.parent / "TELECOMX_LATAM-CHALLENGE-remote" / "TelecomX_Data.json"
    directorio_brutos = RAIZ_REPOSITORIO / "datos" / "brutos"
    directorio_procesados = RAIZ_REPOSITORIO / "datos" / "procesados"
    directorio_informes = RAIZ_REPOSITORIO / "informes"
    directorio_graficos = directorio_informes / "graficos"

    directorio_brutos.mkdir(parents=True, exist_ok=True)
    directorio_procesados.mkdir(parents=True, exist_ok=True)
    directorio_graficos.mkdir(parents=True, exist_ok=True)

    ruta_json_bruto = directorio_brutos / "TelecomX_Datos.json"
    shutil.copy2(ruta_fuente_json, ruta_json_bruto)

    datos_limpios = extraer_y_limpiar_datos(ruta_json_bruto)
    datos_limpios.to_csv(directorio_procesados / "telecomx_limpio.csv", index=False)

    obtener_resumen_preparacion(datos_limpios).to_csv(directorio_informes / "distribucion_clases.csv", index=False)
    obtener_tabla_tasa_categorica(datos_limpios, "tipo_contrato").to_csv(directorio_informes / "tasas_cancelacion_tipo_contrato.csv", index=False)
    obtener_tabla_tasa_categorica(datos_limpios, "metodo_pago").to_csv(directorio_informes / "tasas_cancelacion_metodo_pago.csv", index=False)
    obtener_tabla_tasa_categorica(datos_limpios, "servicio_internet").to_csv(directorio_informes / "tasas_cancelacion_servicio_internet.csv", index=False)
    obtener_resumen_numerico_por_cancelacion(datos_limpios).to_csv(directorio_informes / "resumen_numerico_cancelacion.csv", index=False)

    X, y = preparar_datos_modelado(datos_limpios)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(X, y, proporcion_prueba=0.2)

    metricas_modelos, artefactos = comparar_modelos(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
    metricas_modelos.to_csv(directorio_informes / "metricas_modelos.csv", index=False)

    coeficientes_logit, importancias_bosque = obtener_importancia_variables(artefactos, top_n=12)
    coeficientes_logit.rename("coeficiente").to_csv(directorio_informes / "coeficientes_regresion_logistica.csv")
    importancias_bosque.rename("importancia").to_csv(directorio_informes / "importancia_variables_bosque_aleatorio.csv")

    figura_correlaciones, correlaciones_objetivo = graficar_correlaciones(datos_limpios)
    correlaciones_objetivo.round(4).rename("correlacion").to_csv(directorio_informes / "correlaciones_cancelacion.csv")

    guardar_figura(graficar_distribucion_clases(datos_limpios), directorio_graficos / "distribucion_clases.png")
    guardar_figura(figura_correlaciones, directorio_graficos / "correlaciones_cancelacion.png")
    guardar_figura(graficar_analisis_dirigido(datos_limpios), directorio_graficos / "analisis_dirigido.png")
    guardar_figura(graficar_matrices_confusion(artefactos), directorio_graficos / "matrices_confusion.png")
    guardar_figura(graficar_importancia_variables(coeficientes_logit, importancias_bosque), directorio_graficos / "importancia_variables.png")

    print("Recursos generados correctamente.")
    print(f"Datos procesados: {directorio_procesados / 'telecomx_limpio.csv'}")
    print(f"Informe de metricas: {directorio_informes / 'metricas_modelos.csv'}")


if __name__ == "__main__":
    principal()
