import pandas as pd


df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})
df['weight_cut'] = pd.cut(df['weight'], bins=(60, 75, 80, 95), labels=['light', 'normal', 'heavy'])

# convert to dummy
df = pd.get_dummies(df, dtype='int')

print(df)

data_dict = {
    'currency': [['PLN', 'USD'], ['EUR', 'USD', 'PLN', 'CAD'], ['GBP'], ['JPY', 'CZK', 'HUF'], []]
}
df = pd.DataFrame(data=data_dict)

df['number'] = df['currency'].apply(len)
print(df)

df = pd.DataFrame(data=data_dict)
df['PLN_flag'] = df['currency'].apply(lambda i: 1 if 'PLN' in i else 0)
print(df)


data_dict = {
    'hashtags': [
        '#good#vibes',
        '#hot#summer#holiday',
        '#street#food',
        '#workout'
    ]
}
df = pd.DataFrame(data=data_dict)

df = df['hashtags'].str.split('#', expand=True)
df.drop(columns=0, inplace=True)
df.columns = ['hashtag1', 'hashtag2', 'hashtag3']
df['missing'] = df.isnull().sum(axis=1)
print(df)

data_dict = {
    'investments': [
        '100_000_000',
        '100_000',
        '30_000_000',
        '100_500_000'
    ]
}
df = pd.DataFrame(data=data_dict)
df['investments'] = df['investments'].str.replace('_', '').astype(int)
print(df)

from sklearn.datasets import load_iris

data = load_iris()

print(data.keys())
print(data['feature_names'])
print(data['target_names'])
data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
print(data.shape)
print(target.shape)

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    test_size=0.3
)
print(f'data_train shape: {data_train.shape}')
print(f'target_train shape: {target_train.shape}')
print(f'data_test shape: {data_test.shape}')
print(f'target_test shape: {target_test.shape}')