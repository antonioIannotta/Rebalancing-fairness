import os

import pandas as pd
from fairlearn.reductions import GridSearch, DemographicParity, ExponentiatedGradient
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from metric import disparate_impact, demographic_parity

training_data = pd.read_csv('./dataset/ull_less_columns.csv', sep=',')
test_data = pd.read_csv('./dataset/X_test_less_columns.csv', sep=',')

X = training_data.drop(columns=['income', 'id_student_original', 'id_grade'], inplace=False)
X_test = test_data.drop(columns=['income', 'id_student_original', 'id_grade'], inplace=False)
y = training_data['income']
y_test = pd.read_csv('./dataset/y_test_less_columns.csv', sep=',')

xgb_classifier = XGBClassifier()
rf_classifier = RandomForestClassifier()

xgb_params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

rf_params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

protected_attributes = ['a1', 'capital_island', 'public_private', 'household_income', 'escs']

gs_xgb = GridSearchCV(xgb_classifier, param_grid=xgb_params, cv=10, n_jobs=os.cpu_count())
gs_xgb.fit(X, y)

eg_xgb = ExponentiatedGradient(estimator=gs_xgb.best_estimator_, constraints=DemographicParity())
eg_xgb.fit(X, y, sensitive_features=pd.DataFrame(X, columns=protected_attributes))

# Protected attribute 1 --> a1
# Protected attribute 2 --> capital island
# Protected attribute 3 --> public_private
# Protected attribute 4 --> household_income_q
# Protected attribute 5 --> escs

xgb_pred = eg_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_disparate_impact_a1 = disparate_impact(X_test['a1'].values, xgb_pred)
xgb_disparate_impact_capital_island = disparate_impact(X_test['capital_island'].values, xgb_pred)
xgb_disparate_impact_public_private = disparate_impact(X_test['public_private'].values, xgb_pred)
xgb_disparate_impact_household_income_q = disparate_impact(X_test['household_income_q'].values, xgb_pred)
xgb_disparate_impact_escs = disparate_impact(X_test['escs'].values, xgb_pred)
xgb_demographic_parity_a1 = demographic_parity(X_test['a1'].values, xgb_pred)
xgb_demographic_parity_capital_island = demographic_parity(X_test['capital_island'].values, xgb_pred)
xgb_demographic_parity_public_private = demographic_parity(X_test['public_private'].values, xgb_pred)
xgb_demographic_parity_household_income_q = demographic_parity(X_test['household_income_q'].values, xgb_pred)
xgb_demographic_parity_escs = demographic_parity(X_test['escs'].values, xgb_pred)

file = open('./results/data_augmentation_experiment_results', 'a')
file.write('Model: XGB Classifier \n')
file.write('Accuracy: ' + str(xgb_accuracy) + "\n")
file.write('Disparate impact for a1: ' + str(xgb_disparate_impact_a1) + "\n")
file.write('Disparate impact for capital_island: ' + str(xgb_disparate_impact_capital_island) + "\n")
file.write('Disparate impact for public_private: ' + str(xgb_disparate_impact_public_private) + "\n")
file.write('Disparate impact for household_income_q: ' + str(xgb_disparate_impact_household_income_q) + "\n")
file.write('Disparate impact for escs: ' + str(xgb_demographic_parity_escs) + "\n")
file.write('Demographic parity for a1: ' + str(xgb_demographic_parity_a1) + "\n")
file.write('Demographic parity for capital_island: ' + str(xgb_demographic_parity_capital_island) + "\n")
file.write('Demographic parity for public_private: ' + str(xgb_demographic_parity_public_private) + "\n")
file.write('Demographic parity for household_income_q: ' + str(xgb_demographic_parity_household_income_q) + "\n")
file.write('Demographic parity for escs: ' + str(xgb_demographic_parity_escs) + "\n")

gs_rf = GridSearchCV(rf_classifier, param_grid=rf_params, cv=10, n_jobs=os.cpu_count())
gs_rf.fit(X, y)

rf_pred = gs_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_disparate_impact_a1 = disparate_impact(X_test['a1'].values, rf_pred)
rf_disparate_impact_capital_island = disparate_impact(X_test['capital_island'].values, rf_pred)
rf_disparate_impact_public_private = disparate_impact(X_test['public_private'].values, rf_pred)
rf_disparate_impact_household_income_q = disparate_impact(X_test['household_income_q'].values, rf_pred)
rf_disparate_impact_escs = disparate_impact(X_test['escs'].values, rf_pred)
rf_demographic_parity_a1 = demographic_parity(X_test['a1'].values, rf_pred)
rf_demographic_parity_capital_island = demographic_parity(X_test['capital_island'].values, rf_pred)
rf_demographic_parity_public_private = demographic_parity(X_test['public_private'].values, rf_pred)
rf_demographic_parity_household_income_q = demographic_parity(X_test['household_income_q'].values, rf_pred)
rf_demographic_parity_escs = demographic_parity(X_test['escs'].values, rf_pred)
file = open('./results/data_augmentation_experiment_results', 'a')
file.write('Model: RF Classifier \n')
file.write('Accuracy: ' + str(rf_accuracy) + "\n")
file.write('Disparate impact for a1: ' + str(rf_disparate_impact_a1) + "\n")
file.write('Disparate impact for capital_island: ' + str(rf_disparate_impact_capital_island) + "\n")
file.write('Disparate impact for public_private: ' + str(rf_disparate_impact_public_private) + "\n")
file.write('Disparate impact for household_income_q: ' + str(rf_disparate_impact_household_income_q) + "\n")
file.write('Disparate impact for escs: ' + str(rf_demographic_parity_escs) + "\n")
file.write('Demographic parity for a1: ' + str(rf_demographic_parity_a1) + "\n")
file.write('Demographic parity for capital_island: ' + str(rf_demographic_parity_capital_island) + "\n")
file.write('Demographic parity for public_private: ' + str(rf_demographic_parity_public_private) + "\n")
file.write('Demographic parity for household_income_q: ' + str(rf_demographic_parity_household_income_q) + "\n")
file.write('Demographic parity for escs: ' + str(rf_demographic_parity_escs) + "\n")
