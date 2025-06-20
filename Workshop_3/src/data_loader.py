import os
import pandas as pd

'''
Este módulo contiene funciones para la carga de datos.
Permite centralizar la lógica de lectura de archivos y facilita la reutilización y el mantenimiento del código.
'''

def load_data(data_dir):
    '''
    Carga los archivos de datos de entrenamiento y prueba desde el directorio especificado.
    - Lee los archivos train.csv y test.csv.
    - Devuelve los DataFrames de train, test y un diccionario con información adicional si es necesario.
    '''
    # Construimos las rutas a los archivos de datos
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # Leemos los archivos CSV usando pandas (como abrir un libro para extraer la información relevante)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Diccionario para información adicional, útil para futuras extensiones (por ejemplo, cargar diccionarios de variables)
    data_dict = {}
    
    # Retornamos los DataFrames y el diccionario auxiliar
    return train, test, data_dict