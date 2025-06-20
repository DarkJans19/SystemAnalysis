from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import numpy as np

'''
Este módulo contiene la función para entrenar y validar el modelo de machine learning
Utiliza un Random Forest para clasificación ordinal y evalúa el desempeño usando Quadratic Weighted Kappa (QWK),
la métrica oficial de la competencia
'''

def train_and_evaluate(X, y):
    '''
    Divide los datos en entrenamiento y validación usando un split estratificado
    Entrena un Random Forest y evalúa el desempeño en validación usando QWK
    El split estratificado asegura que todas las clases estén representadas en ambos conjuntos,
    como repartir cartas de una baraja asegurando que cada jugador reciba al menos una de cada palo
    '''
    # Dividimos los datos en train y validation, manteniendo la proporción de clases
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Entrenamos el modelo Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Realizamos predicciones sobre el conjunto de validación
    preds = model.predict(X_val)
    # Calculamos el Quadratic Weighted Kappa para evaluar la calidad de las predicciones
    qwk = cohen_kappa_score(y_val, preds, weights='quadratic')
    # Retornamos el modelo entrenado y la métrica de validación
    return model, qwk