import pandas as pd
import numpy as np

'''
Este módulo contiene funciones para el preprocesamiento de los datos
Incluye imputación de valores faltantes y codificación de variables categóricas
El objetivo es dejar los datos listos para el entrenamiento del modelo, asegurando que no haya valores nulos
y que todas las variables sean numéricas o estén correctamente codificadas
'''

def preprocess(train, test):
    '''
    Realiza el preprocesamiento de los conjuntos de entrenamiento y prueba
    - Imputa valores faltantes en columnas numéricas usando la media del train
    - Codifica la variable de sexo como variable dummy
    - Garantiza que solo se procesen columnas presentes en ambos conjuntos
    '''
    # Seleccionamos solo las columnas numéricas presentes en ambos conjuntos
    num_cols = train.select_dtypes(include=[np.number]).columns
    common_num_cols = [col for col in num_cols if col in test.columns]
    
    # Imputamos valores faltantes con la media del train (simil a rellenar huecos en una pared antes de pintar)
    train[common_num_cols] = train[common_num_cols].fillna(train[common_num_cols].mean())
    test[common_num_cols] = test[common_num_cols].fillna(train[common_num_cols].mean())
    
    # Codificamos la variable de sexo si está presente en ambos conjuntos
    if 'Basic_Demos-Sex' in train.columns and 'Basic_Demos-Sex' in test.columns:
        train['Sex_Label'] = train['Basic_Demos-Sex'].map({0: 'Male', 1: 'Female'})
        test['Sex_Label'] = test['Basic_Demos-Sex'].map({0: 'Male', 1: 'Female'})
    
    # Convertimos la variable de sexo a variable dummy (0/1), eliminando la primera para evitar multicolinealidad
    if 'Sex_Label' in train.columns and 'Sex_Label' in test.columns:
        train = pd.get_dummies(train, columns=['Sex_Label'], drop_first=True)
        test = pd.get_dummies(test, columns=['Sex_Label'], drop_first=True)
    
    # Retornamos los dataframes ya preprocesados y listos para el modelo
    return train, test