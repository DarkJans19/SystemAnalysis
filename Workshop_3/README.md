# Workshop 3: Problematic Internet Use Prediction

## Descripción

Este proyecto implementa un sistema de predicción ordinal para el Severely Impairment Index (SII), siguiendo el flujo de trabajo propuesto en la competencia de Kaggle ["Child Mind Institute - Problematic Internet Use"](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview).

El objetivo es transformar datos crudos de jóvenes en predicciones listas para ser evaluadas, utilizando técnicas de preprocesamiento, ingeniería de features y modelos de machine learning.

---

## Estructura de carpetas

- `data/`: Archivos de datos originales (`train.csv`, `test.csv`, etc.) y diccionario de variables.
- `src/`: Scripts de procesamiento, modelado y utilidades.
- `outputs/`: Resultados y archivos de predicción (`submission.csv`).
- `workshop3_analysis.ipynb`: Análisis exploratorio (opcional).
- `informe_workshop3.md`: Informe formal del Workshop 3.

---

## Instalación de dependencias

Asegúrate de tener Python 3.7+ instalado.  
Instala las dependencias ejecutando:

```
pip install -r requirements.txt
```

---

## Ejecución del pipeline

1. **Coloca los archivos de datos** (`train.csv`, `test.csv`) en la carpeta `data/`.
2. **Ejecuta el script principal** desde la raíz del proyecto:

   ```
   python src/main.py
   ```

3. **Resultado:**  
   Se generará el archivo de predicciones `submission.csv` en la carpeta `outputs/`, listo para subir a Kaggle.

---

## Flujo del sistema

El sistema sigue este flujo:

1. **Carga de datos:**  
   Se leen los archivos de datos y se extraen las variables relevantes.

2. **Preprocesamiento:**  
   - Imputación de valores faltantes (media).
   - Codificación de variables categóricas (por ejemplo, sexo).
   - Selección de features numéricas presentes en ambos conjuntos.

3. **Agrupamiento de la variable objetivo:**  
   La variable `PCIAT-PCIAT_Total` se agrupa en 4 categorías ordinales (0, 1, 2, 3) usando cuartiles, para cumplir con la naturaleza ordinal del problema.

4. **Entrenamiento y validación:**  
   Se entrena un modelo Random Forest y se valida usando la métrica Quadratic Weighted Kappa (QWK), que es la métrica oficial de la competencia.

5. **Predicción y generación de submission:**  
   Se generan las predicciones para el conjunto de test y se guardan en el formato requerido (`id,sii`).

---

## Formato de submission

El archivo `outputs/submission.csv` tendrá el siguiente formato:

```
id,sii
00008ff9,3
000fd460,0
...
```

- `id`: Identificador de cada muestra del test.
- `sii`: Predicción ordinal para SII (valores de 0 a 3).

---

## Requisitos

El archivo `requirements.txt` incluye:

```
pandas
numpy
scikit-learn
jupyter
matplotlib
seaborn
```

---

## Referencias

- [Competencia Kaggle](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview)
- Workshops anteriores (ver documentación adjunta)

---

## Contacto

Para dudas o mejoras, contacta al autor del repositorio o revisa la documentación en los scripts fuente.