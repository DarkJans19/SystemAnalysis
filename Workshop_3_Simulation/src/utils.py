import os
import pandas as pd

'''
This module contains utilities for generating the submission file.
It allows saving predictions in the format required by the Kaggle competition.
'''

def save_submission(test, preds, output_dir):
    '''
    Generates and saves the submission file in CSV format.
    - Includes the 'id' column from the test set and the predictions under the 'sii' column.
    - Creates the output directory if it does not exist.
    - The resulting file is compatible with the Kaggle format.
    '''
    submission = pd.DataFrame({
        'id': test['id'],      # Use the original test identifier
        'sii': preds           # Ordinal prediction for SII
    })
    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)