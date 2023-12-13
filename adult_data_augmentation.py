import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fairness.rebalancing import augment_data

training_data = pd.read_csv('./dataset/adult.data', sep=',')
test_data = pd.read_csv('./dataset/adult.test', sep=',')


categorical_columns = [column for column in training_data.columns if training_data[column].dtype == "O"]
label_encoder = LabelEncoder()
for column in categorical_columns:
    if column == 'income':
        income_array = []
        for val in training_data[column].values:
            if val == '<=50K':
                income_array.append(0)
            else:
                income_array.append(1)
        training_data[column] = income_array
    else:
        training_data[column] = label_encoder.fit_transform(training_data[column])
        test_data[column] = label_encoder.fit_transform(test_data[column])

file = open('./file_augmentation_result.txt', 'a')
file.write("Adult training number of items: \n")
file.write(str(len(training_data)) + "\n")
file.close()

augmented_dataset = augment_data(training_data, ['race', 'sex'], 'income')

file = open('./file_augmentation_result.txt', 'a')
file.write("Adult augmented number of items: \n")
file.write(str(len(augmented_dataset)))
file.close()

augmented_dataset.to_csv('./augmented_dataset.csv', index=False)

