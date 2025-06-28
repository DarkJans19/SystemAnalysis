import pandas as pd

'''
This module contains functions to process and summarize actigraphy data.
It allows extraction of relevant statistics from physical activity signals recorded per subject,
facilitating their integration as features in machine learning models.
'''

def summarize_actigraphy(df, subject_id_col='Subject_ID'):
    '''
    Summarizes actigraphy data by subject.
    - Excludes non-numeric columns and the timestamp column.
    - Calculates descriptive statistics (mean, standard deviation, minimum, and maximum) for each numeric variable.
    - Returns a DataFrame with one row per subject and aggregated statistics columns.
    Analogy: it is like taking each person's activity history and summarizing it into a profile with key values.
    '''
    # Exclude non-numeric columns and timestamp
    exclude_cols = [subject_id_col, 'timestamp']
    num_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    summary = df.groupby(subject_id_col)[num_cols].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary