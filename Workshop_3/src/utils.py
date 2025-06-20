import os
import pandas as pd

'''
Este módulo contiene utilidades para la generación del archivo de submission.
Permite guardar las predicciones en el formato requerido por la competencia de Kaggle.
'''

def save_submission(test, preds, output_dir):
    '''
    Genera y guarda el archivo de submission en formato CSV.
    - Incluye la columna 'id' del test y las predicciones bajo la columna 'sii'
    - Crea el directorio de salida si no existe.
    - El archivo resultante es compatible con el formato de Kaggle
    '''
    submission = pd.DataFrame({
        'id': test['id'],  # Usamos el identificador original del test
        'sii': preds       # Predicción ordinal para SII
    })
    # Creamos el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    # Guardamos el archivo sin índice extra
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)