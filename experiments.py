import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score

dt_c = DecisionTreeClassifier()

data_train = pd.read_csv('./augmented_dataset.csv', sep=',')

target_train = data_train['income']

dt_c.fit(data_train, target_train)

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

predicted = dt_c.predict(data_test)
accuracy_score = accuracy_score(target_test, predicted)

file = open('./results', 'a')
file.write('Accuracy: ' + str(accuracy_score) + "\n")
file.close()

dem_parity_difference = demographic_parity_difference(y_pred=predicted, y_true=target_test,
                                                      sensitive_features=pd.DataFrame(data_test, columns=['sex', 'race']))

file = open('./results', 'a')
file.write('Dem parity difference: ' + str(dem_parity_difference) + "\n")
file.close()
