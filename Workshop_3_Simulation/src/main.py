import os
import pandas as pd
from data_loader import load_data
from features import preprocess
from modtrain import train_and_evaluate
from utils import save_submission

'''
This script implements the complete workflow of an ordinal prediction system for the Severely Impairment Index (SII),
following the architecture diagram. The goal is to transform raw data into predictions
ready to be evaluated in the Kaggle competition "Child Mind Institute - Problematic Internet Use".
'''

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# Load training and test data
train, test, data_dict = load_data(DATA_DIR)

# Remove rows without the target variable
train = train.dropna(subset=['PCIAT-PCIAT_Total'])

train, test = preprocess(train, test)

'''
Group the PCIAT-PCIAT_Total variable into 4 ordinal categories (0, 1, 2, 3) using quartiles.
This converts the problem into an ordinal classification, as required by the competition.
'''
train['SII_group'] = pd.qcut(train['PCIAT-PCIAT_Total'], q=4, labels=[0, 1, 2, 3])
y = train['SII_group'].astype(int)  # Ordinal target variable

# Define columns to exclude from features: the target variable, the identifier, and the new group column
drop_cols = ['PCIAT-PCIAT_Total', 'Subject_ID', 'SII_group']

# Select only numeric columns present in both datasets, excluding drop_cols
# This ensures the model only receives valid and comparable variables between train and test
features = [col for col in train.columns if col not in drop_cols and col in test.columns and train[col].dtype != 'object']
X = train[features]

print(y.value_counts())
print("Selected features:", features)
print("Feature data types:")
print(train[features].dtypes)

'''
Train the Random Forest model and validate using QWK, which is the competition metric.
The use of stratified train_test_split ensures all classes are represented in both sets.
'''
model, qwk = train_and_evaluate(X, y)
print(f'Validación QWK: {qwk:.4f}')

# Generate predictions for the test set using the same features
test_preds = model.predict(test[features])

'''
Ensure the 'id' column is present in test for the submission so that Kaggle accepts the prediction file.
'''
if 'id' not in test.columns:
    original_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test['id'] = original_test['id']

save_submission(test, test_preds, OUTPUT_DIR)