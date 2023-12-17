import pandas as pd
from sklearn.preprocessing import LabelEncoder

adult_train = pd.read_csv('./dataset/adult.data', sep=',')
adult_test = pd.read_csv('./dataset/adult.test', sep=',')

df = pd.concat([adult_train, adult_test], ignore_index=True)

print(df['income'].unique())

categorical_columns = [column for column in df.columns if df[column].dtype == "O"]
label_encoder = LabelEncoder()
for column in categorical_columns:
    if column == 'income':
        income_array = []
        for val in df[column].values:
            if "<=50K" in val:
                income_array.append(0)
            else:
                income_array.append(1)
        df[column] = income_array
    else:
        df[column] = label_encoder.fit_transform(df[column])

print(df['income'].unique())

train = df.iloc[0:len(adult_train)]
test = df.iloc[len(train):]

train.to_csv('./dataset/numerical_adult_train.csv', index=False)
test.to_csv('./dataset/numerical_adult_test.csv', index=False)
