from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=20)
model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)
accuracy = model.score(data_test, target_test)
print(f'Accuracy: {accuracy:.4f}')

target_pred = model.predict(data_test)
print(target_pred)

cm = confusion_matrix(target_test, target_pred)
print(cm)
print(classification_report(target_test, target_pred))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')

le = LabelEncoder()
df['bought'] = le.fit_transform(df['bought'])
print(df)

data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')

encoder = OneHotEncoder(sparse=False)
encoder.fit(df[['size']])
print(encoder.transform(df[['size']]))
print(encoder.categories_)



from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()
print(raw_data['DESCR'])

data = raw_data['data']
target = raw_data['target']

print(data[:3])

all_data = np.c_[data, target]
print(all_data[:3])

df = pd.DataFrame(
    data=all_data,
    columns=list(raw_data['feature_names']) + ['target']
)
print(df.head())