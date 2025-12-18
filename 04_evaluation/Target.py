# Script de Análisis Exploratorio de Datos (EDA)
# Utilizado para analizar la distribución de la variable objetivo (TARGET)
# No forma parte del pipeline productivo


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("Data/processed/dataset_merged.parquet")

df["TARGET"].value_counts().plot(
    kind="bar",
    title="Distribución de la variable objetivo (TARGET)"
)

plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()
