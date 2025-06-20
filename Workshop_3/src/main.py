import os
import pandas as pd
from data_loader import load_data
from features import preprocess
from modtrain import train_and_evaluate
from utils import save_submission

'''
Este script implementa el flujo completo de un sistema de predicción ordinal para el Severely Impairment Index (SII),
siguiendo el esquema planteado en el diagrama de arquitectura. El objetivo es transformar datos crudos en predicciones
listas para ser evaluadas en la competencia de Kaggle "Child Mind Institute - Problematic Internet Use".
'''

# Definimos los directorios de entrada y salida de datos de forma relativa para portabilidad
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# Cargamos los datos de entrenamiento y prueba usando una función modular
train, test, data_dict = load_data(DATA_DIR)

# Eliminamos filas sin la variable objetivo, ya que no aportan al entrenamiento
train = train.dropna(subset=['PCIAT-PCIAT_Total'])

# Preprocesamos los datos: imputación de valores faltantes y codificación de variables categóricas
train, test = preprocess(train, test)

'''
Agrupamos la variable PCIAT-PCIAT_Total en 4 categorías ordinales (0, 1, 2, 3) usando cuartiles
Esto convierte el problema en una clasificación ordinal, como exige la competencia
'''
train['SII_group'] = pd.qcut(train['PCIAT-PCIAT_Total'], q=4, labels=[0, 1, 2, 3])
y = train['SII_group'].astype(int)  # Variable objetivo ordinal

# Definimos las columnas a excluir de las features: la variable objetivo, el identificador y la nueva columna de grupo
drop_cols = ['PCIAT-PCIAT_Total', 'Subject_ID', 'SII_group']

# Seleccionamos solo las columnas numéricas presentes en ambos conjuntos, excluyendo las de drop_cols
# Esto asegura que el modelo solo reciba variables válidas y comparables entre train y test
features = [col for col in train.columns if col not in drop_cols and col in test.columns and train[col].dtype != 'object']
X = train[features]

# Mostramos la distribución de clases y las features seleccionadas para depuración y transparencia
print(y.value_counts())
print("Features seleccionadas:", features)
print("Tipos de datos de las features:")
print(train[features].dtypes)

'''
Entrenamos el modelo Random Forest y validamos usando QWK que es la métrica de la competencia.
El uso de train_test_split estratificado garantiza que todas las clases estén representadas en ambos conjuntos.
'''
model, qwk = train_and_evaluate(X, y)
print(f'Validación QWK: {qwk:.4f}')

# Generamos predicciones para el conjunto de test usando las mismas features
test_preds = model.predict(test[features])

'''
Nos aseguramos de que la columna 'id' esté presente en test para la submission para que Kaggle acepte el archivo de predicción.
'''
if 'id' not in test.columns:
    original_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test['id'] = original_test['id']

# Guardamos el archivo de submission en el formato requerido
save_submission(test, test_preds, OUTPUT_DIR)