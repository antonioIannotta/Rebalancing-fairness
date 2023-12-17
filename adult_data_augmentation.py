import pandas as pd
from fairness.rebalancing import augment_data

training_data = pd.read_csv('./dataset/numerical_adult_train.csv', sep=',')

augmented_dataset = augment_data(training_data, ['race', 'sex'], 'income')

file = open('./file_augmentation_result.txt', 'a')
file.write("Adult augmented number of items: \n")
file.write(str(len(augmented_dataset)))
file.close()

augmented_dataset.to_csv('./augmented_dataset.csv', index=False)

