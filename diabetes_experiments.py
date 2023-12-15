import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from metric import *

from fairness.rebalancing import augment_data

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

file = open('./diabetes_results', 'a')
file.write('Training rows: ' + str(len(X_train)) + "\n")
file.close()

X_train['Diabetes_binary'] = y_train

augmented_dataset = augment_data(X_train, ['Education', 'Sex'], 'Diabetes_binary')

file = open('./diabetes_results', 'a')
file.write("Diabetes augmented number of items: \n")
file.write(str(len(augmented_dataset)) + "\n")
file.close()

augmented_dataset.to_csv('./diabetes_augmented_dataset.csv', index=False)

model = RandomForestClassifier()

train = pd.read_csv('./diabetes_augmented_dataset.csv', sep=',')
data_train = train.drop(columns=['Diabetes_binary'], inplace=False)
target_train = train['Diabetes_binary']

model.fit(data_train, target_train)

predicted = model.predict(X_test)
accuracy_score = accuracy_score(y_test, predicted)

file = open('./diabetes_results', 'a')
file.write('Accuracy: ' + str(accuracy_score) + "\n")
file.close()

dem_parity_sex = demographic_parity(X_test['Sex'].values, predicted)
dem_parity_education = demographic_parity(X_test['Education'].values, predicted)
equalized_odds_education = equalized_odds(X_test['Education'].values, y_true=y_test, y_pred=predicted)
equalized_odds_sex = equalized_odds(X_test['Sex'].values, y_true=y_test, y_pred=predicted)
disparate_impact_education = disparate_impact(X_test['Education'].values, predicted)
disparate_impact_sex = disparate_impact(X_test['Sex'].values, predicted)

file = open('./diabetes_results', 'a')
file.write('Demographic parity sex: ' + str(dem_parity_sex) + "\n")
file.write('Demographic parity education: ' + str(dem_parity_education) + "\n")
file.write('Disparate impact sex: ' + str(disparate_impact_sex) + "\n")
file.write('Disparate impact education: ' + str(disparate_impact_education) + "\n")
file.write('Equalized odds sex: ' + str(equalized_odds_sex) + "\n")
file.write('Equalized odds education: ' + str(equalized_odds_education) + "\n")
file.close()

