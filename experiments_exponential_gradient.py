import os

import numpy as np
from fairlearn.preprocessing import CorrelationRemover
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from fairlearn.reductions import DemographicParity

from sklearn.metrics import accuracy_score
from metric import demographic_parity, disparate_impact

training_data = pd.read_csv('./dataset/numerical_adult_train.csv', sep=',')
test_data = pd.read_csv('./dataset/numerical_adult_test.csv', sep=',')

X = training_data.drop(columns=['income'], inplace=False)
X_test = test_data.drop(columns=['income'], inplace=False)
y = training_data['income']
y_test = test_data['income']

xgb_params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

sgd_params = {
    'loss': ['log'],
    'penalty': ['elasticnet'],
    'n_iter': [5],
    'alpha': np.logspace(-4, 4, 10),
    'l1_ratio': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2]
}

rf_params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

