import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fairness.rebalancing import augment_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from metric import *

data = pd.read_csv('./dataset/german.data', sep=' ')

categorical_columns = [column for column in data.columns if data[column].dtype == "O" or column == 'output']
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

X = data.drop(columns=['output'], inplace=False)
print(X)
y = data['output']
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

file = open('./german_results', 'a')
file.write('Training rows: ' + str(len(X_train)) + "\n")

X_train['output'] = y_train

augmented_dataset = augment_data(X_train, ['a9', 'a20'], 'output')

file = open('./german_results', 'a')
file.write("German augmented number of items: \n")
file.write(str(len(augmented_dataset)) + "\n")
file.close()

augmented_dataset.to_csv('./german_augmented_dataset.csv', index=False)

model = RandomForestClassifier()

train = pd.read_csv('./german_augmented_dataset.csv', sep=',')
data_train = train.drop(columns=['output'], inplace=False)
target_train = train['output']

model.fit(data_train, target_train)

predicted = model.predict(X_test)
accuracy_score = accuracy_score(y_test, predicted)

file = open('./german_results', 'a')
file.write('Accuracy: ' + str(accuracy_score) + "\n")
file.close()

dem_parity_sex = demographic_parity(X_test['a9'].values, predicted)
dem_parity_foreign_worker = demographic_parity(X_test['a20'].values, predicted)
equalized_odds_foreign_worker = equalized_odds(X_test['a20'].values, y_true=y_test, y_pred=predicted)
equalized_odds_sex = equalized_odds(X_test['a9'].values, y_true=y_test, y_pred=predicted)
disparate_impact_foreign_worker = disparate_impact(X_test['a20'].values, predicted)
disparate_impact_sex = disparate_impact(X_test['a9'].values, predicted)

file = open('./german_results', 'a')
file.write('Demographic parity sex: ' + str(dem_parity_sex) + "\n")
file.write('Demographic parity race: ' + str(dem_parity_foreign_worker) + "\n")
file.write('Disparate impact sex: ' + str(disparate_impact_sex) + "\n")
file.write('Disparate impact race: ' + str(disparate_impact_foreign_worker) + "\n")
file.write('Equalized odds sex: ' + str(equalized_odds_sex) + "\n")
file.write('Equalized odds race: ' + str(equalized_odds_foreign_worker) + "\n")
file.close()
