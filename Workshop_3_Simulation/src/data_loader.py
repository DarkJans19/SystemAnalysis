import os
import pandas as pd

'''
This module contains functions for data loading.
It centralizes the logic for reading files and facilitates code reuse and maintenance.
'''

def load_data(data_dir):
    '''
    Loads the training and test data files from the specified directory.
    - Reads train.csv and test.csv files.
    - Returns the train and test DataFrames, and a dictionary for additional information if needed.
    '''

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # Read CSV files using pandas
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    data_dict = {}
    
    # Return the DataFrames and the auxiliary dictionary
    return train, test, data_dict