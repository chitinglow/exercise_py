import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=40)

print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')

for name, array in zip(['target', 'y_train', 'y_test'], [target, y_train, y_test]):
    print(f'{name.ljust(7)}:{np.unique(array, return_counts=True)[1] / len(array)}')

import numpy as np
import pandas as pd

df = pd.DataFrame({'years': [1, 2, 3, 4, 5, 6],
                   'salary': [4000, 4250, 4500, 4750, 5000, 5250]})

m = len(df)

X1 = df['years'].values
Y = df['salary'].values

X1 = X1.reshape(m, 1)
bias = np.ones((m, 1))
X = np.append(bias, X1, axis=1)

coefs = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
print(f'Linear regression: {coefs[0]:.2f} + {coefs[1]:.2f}x')



df = pd.DataFrame({'years': [1, 2, 3, 4, 5, 6],
                   'salary': [4000, 4250, 4500, 4750, 5000, 5250]})

reg = LinearRegression().fit(df[['years']], df[['salary']])
print(f'Linear regression: {reg.intercept_[0]:.2f} '
      f'+ {reg.coef_[0][0]:.2f}x')


df = pd.read_csv('data.csv')
data = df[['variable']]
target = df['target']

model = LinearRegression()
model.fit(data, target)
print(f'{model.score(data, target):.4f}')

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)

df_poly = poly.fit_transform(df)
print(df_poly)

import numpy as np
import pandas as pd

predictions = pd.read_csv('predictions.csv')
def mean_absolute_error(y_true, y_pred):
    return abs(y_true - y_pred).sum() / len(y_true)
mae = mean_absolute_error(predictions['y_true'], predictions['y_pred'])
print(f'MAE = {mae:.4f}')

predictions = pd.read_csv('predictions.csv')


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_true)


mse = mean_squared_error(predictions['y_true'], predictions['y_pred'])
print(f'MSE = {mse:.4f}')