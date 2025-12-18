
README
------------------------------------------------------------------------------------
Nombre del Proyecto: Predicción de Riesgo de Incumplimiento de Crédito

Integrantes:

Marcelo Mancilla,
Valentina Morales

Asignatura: Machinne Lerning
---------------------------------------------------------------------------------------------
0. Descripción general del proyecto

En este proyecto se desarrolla un proceso completo de análisis de datos y modelado de Machine Learning, 
siguiendo la metodología CRISP-DM. 

Problema abordado: El problema principal que se busca resolver es la segmentación de clientes a partir de sus características socioeconómicas
y financieras, con el fin de identificar grupos con comportamientos similares que apoyen la toma de decisiones de negocio.

Enfoque de la solución: Para abordar este problema, se analiza un conjunto de datos, se realiza una preparación adecuada de la información, 
se aplica un modelo de aprendizaje automático y se evalúan los resultados obtenidos.

El proyecto está organizado por fases, lo que permite comprender claramente cada etapa del proceso y facilita su mantención y reutilización.

____________________________________________________________________________________________________________________________

1. Técnica utilizada y justificación
   
En este proyecto se utilizaron técnicas de aprendizaje supervisado y no supervisado, 
con el objetivo de analizar los datos desde distintos enfoques y obtener información complementaria.

Por una parte, se aplicó aprendizaje no supervisado mediante K-Means, 
cuyo propósito fue realizar la segmentación de clientes a partir de sus características socioeconómicas y financieras,
permitiendo identificar grupos con comportamientos similares sin contar con etiquetas previas.

Por otra parte, se incorporó un modelo de aprendizaje supervisado,
el cual permitió evaluar relaciones entre las variables y apoyar el análisis predictivo, utilizando datos previamente etiquetados.

La combinación de ambos enfoques permitió un análisis más completo del problema, aprovechando las ventajas del aprendizaje no supervisado para descubrir patrones
y del aprendizaje supervisado para validar y reforzar los resultados obtenidos.

____________________________________________________________________________________________________________________________


2. Instrucciones de ejecución

A continuación, se detallan los pasos necesarios para ejecutar el proyecto de manera reproducible:

1. Clonar o descargar el repositorio del proyecto en el equipo local.
2. Verificar que se cuente con Python instalado (versión 3.8 o superior).
3. Instalar las dependencias necesarias ejecutando: pip install -r requirements.txt
4. Ejecutar los scripts o notebooks siguiendo el orden de las fases CRISP-DM:
   *EDA/eda.ipynb: análisis exploratorio de los datos.
   *preparacion/preparacion_datos.py: limpieza y transformación de los datos.
   *modelado/: ejecución de los modelos supervisado y no supervisado
   *evaluacion/: análisis y evaluación de los resultados obtenidos.
   *deployment/ (opcional): generación de predicciones o uso del modelo entrenado.
Siguiendo estos pasos, es posible reproducir el flujo completo del proyecto desde el análisis inicial hasta la obtención de resultados.

_________________________________________________________________________________________________________________________________

3. Resultados

El análisis exploratorio realizado mediante el modelo no supervisado K-Means permitió identificar distintos grupos de observaciones con características similares, 
lo que facilitó la comprensión de patrones relevantes presentes en los datos.

Posteriormente, el modelo de aprendizaje supervisado logró capturar la relación existente entre las variables de entrada y la variable objetivo,
obteniendo un desempeño satisfactorio. Las métricas de evaluación evidencian que el modelo es capaz de realizar predicciones consistentes, 
aportando información útil y relevante para el contexto del proyecto.

____________________________________________________________________________________________________________

4. Conclusiones y uso futuro

A partir del desarrollo del proyecto, se concluye que la aplicación de técnicas de aprendizaje supervisado y no supervisado permitió abordar de manera efectiva el problema planteado,
entregando información relevante sobre los patrones presentes en los datos.

La segmentación obtenida mediante K-Means permitió identificar grupos diferenciados de clientes, facilitando una mejor comprensión de sus características y comportamientos. 
Asimismo, el uso de modelos supervisados complementó el análisis, aportando validación y apoyo a las conclusiones obtenidas.

Como uso futuro, este proyecto podría ser incorporado en un entorno productivo para apoyar la toma de decisiones, por ejemplo, en estrategias de segmentación,
personalización de servicios o análisis predictivo. Además, el modelo puede ser mejorado incorporando nuevas variables, ajustando hiperparámetros o evaluando técnicas adicionales 
de Machine Learning que permitan optimizar los resultados.



