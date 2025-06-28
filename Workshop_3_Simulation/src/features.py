import pandas as pd
import numpy as np

'''
This module contains functions for data preprocessing.
It includes missing value imputation and categorical variable encoding.
The goal is to prepare the data for model training, ensuring there are no null values
and that all variables are numeric or properly encoded.
'''

def preprocess(train, test):
    '''
    Preprocesses the training and test datasets.
    - Imputes missing values in numeric columns using the mean from the training set.
    - Encodes the sex variable as a dummy variable.
    - Ensures only columns present in both datasets are processed.
    '''
    # Select only numeric columns present in both datasets
    num_cols = train.select_dtypes(include=[np.number]).columns
    common_num_cols = [col for col in num_cols if col in test.columns]
    
    # Impute missing values with the mean from the training set
    train[common_num_cols] = train[common_num_cols].fillna(train[common_num_cols].median())
    test[common_num_cols] = test[common_num_cols].fillna(train[common_num_cols].median())
    
    # Encode the sex variable if present in both datasets
    if 'Basic_Demos-Sex' in train.columns and 'Basic_Demos-Sex' in test.columns:
        train['Sex_Label'] = train['Basic_Demos-Sex'].map({0: 'Male', 1: 'Female'})
        test['Sex_Label'] = test['Basic_Demos-Sex'].map({0: 'Male', 1: 'Female'})
    
    # Convert the sex variable to a dummy variable (0/1), dropping the first to avoid multicollinearity
    if 'Sex_Label' in train.columns and 'Sex_Label' in test.columns:
        train = pd.get_dummies(train, columns=['Sex_Label'], drop_first=True)
        test = pd.get_dummies(test, columns=['Sex_Label'], drop_first=True)
    
    return train, test