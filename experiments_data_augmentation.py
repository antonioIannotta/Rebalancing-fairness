import os

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from metric import disparate_impact, demographic_parity

training_data = pd.read_csv('./dataset/augmented_dataset.csv', sep=',')
test_data = pd.read_csv('./dataset/numerical_adult_test.csv', sep=',')

X = training_data.drop(columns=['income'], inplace=False)
X_test = test_data.drop(columns=['income'], inplace=False)
y = training_data['income']
y_test = test_data['income']

xgb_classifier = XGBClassifier()
sgd_classifier = SGDClassifier()
rf_classifier = RandomForestClassifier()

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


gs_xgb = GridSearchCV(xgb_classifier, param_grid=xgb_params, cv=10, n_jobs=os.cpu_count())
gs_xgb.fit(X, y)

xgb_pred = gs_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_disparate_impact_race = disparate_impact(X_test['race'].values, xgb_pred)
xgb_disparate_impact_sex = disparate_impact(X_test['sex'].values, xgb_pred)
xgb_demographic_parity_race = demographic_parity(X_test['race'].values, xgb_pred)
xgb_demographic_parity_sex = demographic_parity(X_test['sex'].values, xgb_pred)

file = open('./results/data_augmentation_experiment_results', 'a')
file.write('Model: XGB Classifier \n')
file.write('Accuracy: ' + str(xgb_accuracy) + "\n")
file.write('Disparate impact for race: ' + str(xgb_disparate_impact_race) + "\n")
file.write('Disparate impact for sex: ' + str(xgb_disparate_impact_sex) + "\n")
file.write('Demographic parity for race: ' + str(xgb_demographic_parity_race) + "\n")
file.write('Demographic parity for sex: ' + str(xgb_demographic_parity_sex) + "\n")

gs_sgd = GridSearchCV(sgd_classifier, param_grid=sgd_params, cv=10, n_jobs=os.cpu_count())
gs_sgd.fit(X, y)

sgd_pred = gs_sgd.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_pred)
sgd_disparate_impact_race = disparate_impact(X_test['race'].values, sgd_pred)
sgd_disparate_impact_sex = disparate_impact(X_test['sex'].values, sgd_pred)
sgd_demographic_parity_race = demographic_parity(X_test['race'].values, sgd_pred)
sgd_demographic_parity_sex = demographic_parity(X_test['sex'].values, sgd_pred)

file = open('./results/data_augmentation_experiment_results', 'a')
file.write('Model: SGD Classifier \n')
file.write('Accuracy: ' + str(sgd_accuracy) + "\n")
file.write('Disparate impact for race: ' + str(sgd_disparate_impact_race) + "\n")
file.write('Disparate impact for sex: ' + str(sgd_disparate_impact_sex) + "\n")
file.write('Demographic parity for race: ' + str(sgd_demographic_parity_race) + "\n")
file.write('Demographic parity for sex: ' + str(sgd_demographic_parity_sex) + "\n")


gs_rf = GridSearchCV(rf_classifier, param_grid=rf_params, cv=10, n_jobs=os.cpu_count())
gs_rf.fit(X, y)

rf_pred = gs_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_disparate_impact_race = disparate_impact(X_test['race'].values, rf_pred)
rf_disparate_impact_sex = disparate_impact(X_test['sex'].values, rf_pred)
rf_demographic_parity_race = demographic_parity(X_test['race'].values, rf_pred)
rf_demographic_parity_sex = demographic_parity(X_test['sex'].values, rf_pred)

file = open('./results/data_augmentation_experiment_results', 'a')
file.write('Model: RF Classifier \n')
file.write('Accuracy: ' + str(rf_accuracy) + "\n")
file.write('Disparate impact for race: ' + str(rf_disparate_impact_race) + "\n")
file.write('Disparate impact for sex: ' + str(rf_disparate_impact_sex) + "\n")
file.write('Demographic parity for race: ' + str(rf_demographic_parity_race) + "\n")
file.write('Demographic parity for sex: '+ str(rf_demographic_parity_sex) + "\n")
