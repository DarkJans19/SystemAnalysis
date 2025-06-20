import pandas as pd

'''
Este módulo contiene funciones para procesar y resumir datos de actigrafía
Permite extraer estadísticas relevantes de las señales de actividad física registradas por sujeto,
facilitando su integración como features en modelos de machine learning
'''

def summarize_actigraphy(df, subject_id_col='Subject_ID'):
    '''
    Resume los datos de actigrafía por sujeto.
    - Excluye columnas no numéricas y la columna de timestamp.
    - Calcula estadísticas descriptivas (media, desviación estándar, mínimo y máximo) para cada variable numérica.
    - Devuelve un DataFrame con una fila por sujeto y columnas de estadísticas agregadas.
    Simil: es como tomar el historial de actividad de cada persona y resumirlo en una ficha con sus valores clave.
    '''
    # Excluye columnas no numéricas ni timestamp
    exclude_cols = [subject_id_col, 'timestamp']
    num_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Calcula estadísticas por sujeto (media, std, min, max)
    summary = df.groupby(subject_id_col)[num_cols].agg(['mean', 'std', 'min', 'max'])
    # Aplana el MultiIndex de columnas para facilitar su uso posterior
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary