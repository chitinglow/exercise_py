import sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)
print(np.round(df.isnull().sum() / len(df), 2))

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['weight']] = imputer.fit_transform(df[['weight']])
print(imputer.statistics_[0])

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value=99.0
)
df[['price']] = imputer.fit_transform(df[['price']])
print(df)

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='most_frequent'
)
df[['size']] = imputer.fit_transform(df[['size']])
print(df)

df = pd.DataFrame(data=data)
print(df[~df['weight'].isnull()].mean(numeric_only=True))

df_object = df.select_dtypes(include=['object']).fillna('empty')
print(df_object)

df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})
df['weight_cut'] = pd.cut(df['weight'], bins=3)
print(df)

df['weight_cut'] = pd.cut(df['weight'], bins=(60, 75, 80, 95))
print(df)

df['weight_cut'] = pd.cut(
    df['weight'],
    bins=(60, 75, 80, 95),
    labels=['light', 'normal', 'heavy']
)
print(df)