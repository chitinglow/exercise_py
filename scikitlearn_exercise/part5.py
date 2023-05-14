import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame(data=np.random.randn(10), columns=['var1'])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


df['var1_sigmoid'] = df['var1'].apply(sigmoid)
print(df)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)
df = pd.read_csv('data.csv')

scaler = StandardScaler()
scaler.fit(df)

df_scaled = scaler.transform(df)
print(df_scaled)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])
print(X_test[:5])


def entropy(x):
    return np.round(-np.sum(x * np.log2(x)), 4)


data_dict = {
    'val_1': np.arange(0.01, 1.0, 0.1),
    'val_2': 1 - np.arange(0.01, 1.0, 0.1)
}
df = pd.DataFrame(data_dict)

df['entropy'] = [
    entropy([row[1][0], row[1][1]])
    for row in df.iterrows()
]
print(df)

from sklearn.metrics import accuracy_score

df = pd.read_csv('predictions.csv')

acc = accuracy_score(df['y_true'], df['y_pred'])
print(f"Accuracy: {acc:.4f}")

from sklearn.metrics import confusion_matrix

df = pd.read_csv('predictions.csv')

print(confusion_matrix(df['y_true'], df['y_pred']))

import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
acc = classifier.score(X_test, y_test)
print(f'Accuracy: {acc:.4f}')

import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)

classifier = DecisionTreeClassifier()

params = {'max_depth': np.arange(1, 10),
          'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}

grid_search = GridSearchCV(
    classifier,
    param_grid=params,
    scoring='accuracy',
    cv=5
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)