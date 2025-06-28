from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

'''
This module contains the function to train and validate the machine learning model.
It uses a Random Forest for ordinal classification and evaluates performance using Quadratic Weighted Kappa (QWK),
the official metric of the competition.
'''

def train_and_evaluate(X, y):
    '''
    Splits the data into training and validation sets using stratified split.
    Trains a Random Forest and evaluates performance on the validation set using QWK.
    The stratified split ensures all classes are represented in both sets,
    similar to dealing cards so each player gets at least one of each suit.
    '''
    # Split the data into train and validation sets, maintaining class proportions
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    # Calculate the Quadratic Weighted Kappa to evaluate prediction quality
    qwk = cohen_kappa_score(y_val, preds, weights='quadratic')
    return model, qwk