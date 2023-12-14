import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from xgboost import XGBClassifier

from metric import *

xgb_clf = XGBClassifier()

data_train = pd.read_csv('./augmented_dataset.csv', sep=',')
target_train = data_train['income']

xgb_clf.fit(data_train, target_train)

data_test = pd.read_csv('./dataset/adult.test', sep=',')

categorical_columns = [column for column in data_test.columns if data_test[column].dtype == "O"]
label_encoder = LabelEncoder()
for column in categorical_columns:
    if column == 'income':
        income_array = []
        for val in data_test[column].values:
            if val == '<=50K':
                income_array.append(0)
            else:
                income_array.append(1)
        data_test[column] = income_array
    else:
        data_test[column] = label_encoder.fit_transform(data_test[column])

target_test = data_test['income']

predicted = xgb_clf.predict(data_test)
accuracy_score = accuracy_score(target_test, predicted)

file = open('./results', 'a')
file.write('Accuracy: ' + str(accuracy_score) + "\n")
file.close()

dem_parity_sex = demographic_parity(data_test['sex'].values, predicted)
dem_parity_race = demographic_parity(data_test['race'].values, predicted)
equalized_odds_race = equalized_odds(data_test['race'].values, y_true=target_test, y_pred=predicted)
equalized_odds_sex = equalized_odds(data_test['sex'].values, y_true=target_test, y_pred=predicted)
disparate_impact_race = disparate_impact(data_test['race'].values, predicted)
disparate_impact_sex = disparate_impact(data_test['sex'].values, predicted)

file = open('./results', 'a')
file.write('Demographic parity sex: ' + str(dem_parity_sex) + "\n")
file.write('Demographic parity race: ' + str(dem_parity_race) + "\n")
file.write('Disparate impact sex: ' + str(disparate_impact_sex) + "\n")
file.write('Disparate impact race: ' + str(disparate_impact_race) + "\n")
file.write('Equalized odds sex: ' + str(equalized_odds_sex) + "\n")
file.write('Equalized odds race: ' + str(equalized_odds_race) + "\n")
file.close()
