import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

df = pd.read_csv('clusters.csv')
wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))
print(wcss)

np.random.seed(42)
df = pd.read_csv('clusters.csv')

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))
print(3)


cluster = AgglomerativeClustering(n_clusters=2)
cluster.fit_predict(df)

df = pd.DataFrame(df, columns=['x1', 'x2'])
df['cluster'] = cluster.labels_

print(df.head(10))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('clusters.csv')

cluster = AgglomerativeClustering(
    n_clusters=2,
    affinity='manhattan',
    linkage='complete'
)
cluster.fit_predict(df)

df = pd.DataFrame(df, columns=['x1', 'x2'])
df['cluster'] = cluster.labels_

print(df.head(10))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.set_printoptions(
    precision=8,
    suppress=True,
    edgeitems=5,
    linewidth=200
)
np.random.seed(42)
df = pd.read_csv('pca.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

eig_vals, eig_vecs = np.linalg.eig(np.cov(X_std, rowvar=False))
eig_pairs = [
    (np.abs(eig_vals[i]), eig_vecs[:, i])
    for i in range(len(eig_vals))
]
eig_pairs.sort(reverse=True)

W = np.hstack((
    eig_pairs[0][1].reshape(3, 1),
    eig_pairs[1][1].reshape(3, 1)
))
X_pca = X_std.dot(W)
print(X_pca[:10])

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.set_printoptions(
    precision=8,
    suppress=True,
    edgeitems=5,
    linewidth=200
)
np.random.seed(42)
df = pd.read_csv('pca.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

eig_vals, eig_vecs = np.linalg.eig(np.cov(X_std, rowvar=False))
eig_pairs = [
    (np.abs(eig_vals[i]), eig_vecs[:, i])
    for i in range(len(eig_vals))
]
eig_pairs.sort(reverse=True)

W = np.hstack((
    eig_pairs[0][1].reshape(3, 1),
    eig_pairs[1][1].reshape(3, 1)
))
X_pca = X_std.dot(W)

df_pca = pd.DataFrame(data=X_pca, columns=['pca_1', 'pca_2'])
df_pca['class'] = df['class']
df_pca['pca_2'] = - df_pca['pca_2']
print(df_pca.head(10))

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('pca.csv')
data = df.values

scaler = StandardScaler()
data_std = scaler.fit_transform(data)

pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_std)

results = pd.DataFrame(
    data={'explained_variance_ratio': pca.explained_variance_ratio_}
)
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('pca.csv')
data = df.values

scaler = StandardScaler()
data_std = scaler.fit_transform(data)

pca = PCA(n_components=0.95)
data_pca = pca.fit_transform(data_std)

print(f'Number of components: {pca.n_components_}')

import numpy as np
import pandas as pd

data = {
    "products": [
        "bread eggs",
        "bread eggs milk",
        "milk cheese",
        "bread butter cheese",
        "eggs milk",
        "bread milk butter cheese",
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

trans_encoded = np.zeros(
    (len(transactions), len(products)),
    dtype='int8'
)

for row in zip(range(len(transactions)), trans_encoded, expanded.values):
    for idx, product in enumerate(products):
        if product in row[2]:
            trans_encoded[row[0], idx] = 1

trans_encoded_df = pd.DataFrame(trans_encoded, columns=products)

sup_butter_bread = len(trans_encoded_df.query("butter == 1 and bread == 1")) / len(trans_encoded_df)
sup_butter_milk = len(trans_encoded_df.query("butter == 1 and milk == 1")) / len(trans_encoded_df)

print(f'support(butter, bread) = {sup_butter_bread:.4f}')
print(f'support(butter, milk) = {sup_butter_milk:.4f}')

import numpy as np
import pandas as pd

data = {
    "products": [
        "bread eggs",
        "bread eggs milk",
        "milk cheese",
        "bread butter cheese",
        "eggs milk",
        "bread milk butter cheese",
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

trans_encoded = np.zeros((len(transactions), len(products)), dtype='int8')

for row in zip(range(len(transactions)), trans_encoded, expanded.values):
    for idx, product in enumerate(products):
        if product in row[2]:
            trans_encoded[row[0], idx] = 1

trans_encoded_df = pd.DataFrame(trans_encoded, columns=products)

conf_cheese_bread = len(trans_encoded_df.query("cheese == 1 and bread == 1")) \
                    / len(trans_encoded_df.query("cheese== 1"))
conf_butter_cheese = len(trans_encoded_df.query("butter == 1 and cheese == 1")) \
                     / len(trans_encoded_df.query("butter== 1"))

print(f'conf(cheese, bread) = {conf_cheese_bread:.4f}')
print(f'conf(butter, cheese) = {conf_butter_cheese:.4f}')

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
df = pd.read_csv('blobs.csv')
data = df.values

lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(data)

df['lof'] = y_pred
print(df.head(10))

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
df = pd.read_csv('blobs.csv')
data = df.values

lof = LocalOutlierFactor(n_neighbors=30)
y_pred = lof.fit_predict(data)

df['lof'] = y_pred
print(df['lof'].value_counts())

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


np.random.seed(42)

df = pd.read_csv('factory.csv')

outlier = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
outlier.fit(df)
df['outlier_flag'] = outlier.predict(df)
print(df.head(10))
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

np.random.seed(42)
df = pd.read_csv('factory.csv')

outlier = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
outlier.fit(df)
df['outlier_flag'] = outlier.predict(df)
print(df['outlier_flag'].value_counts())


import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


np.random.seed(42)
data, target = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
data, target = load_digits(return_X_y=True)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
acc = neigh.score(X_test, y_test)
print(f'KNN accuracy: {acc:.4f}')


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Logistic Regression accuracy: {acc:.4f}")

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

print(data_train['text'][1])

data_train = data_train['text'].tolist()
print(len(data_train))

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = CountVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)
print(data_train_vectorized.shape)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = CountVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)

classifier = MultinomialNB()
classifier.fit(data_train_vectorized, target_train)

docs = [
    'The graphic designer requires a good processor to work',
    'Flights into space'
]
data_new = vectorizer.transform(docs)

data_pred = classifier.predict(data_new)

for doc, category in zip(docs, data_pred):
    print(f'\'{doc}\' => {categories[category]}')


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = TfidfVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)
print(data_train_vectorized.shape)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = TfidfVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)

classifier = MultinomialNB()
classifier.fit(data_train_vectorized, target_train)

docs = [
    'The graphic designer requires a good processor to work',
    'Flights into space'
]
data_new = vectorizer.transform(docs)

data_pred = classifier.predict(data_new)

for doc, category in zip(docs, data_pred):
    print(f'\'{doc}\' => {categories[category]}')

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target']
)
print(df.head())

print(df.corr()['target'].sort_values(ascending=False)[1:])

data = df.copy()
target = data.pop('target')

print(data.head())
print()
print(target.head())

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=42
)

print(f'data_train shape: {data_train.shape}')
print(f'target_train shape: {target_train.shape}')
print(f'data_test shape: {data_test.shape}')
print(f'target_test shape: {target_test.shape}')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(data_train, target_train)
print(f'R^2 score: {regressor.score(data_test, target_test):.4f}')

target_pred = regressor.predict(data_test)
print(target_pred)

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target']
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=42
)

regressor = LinearRegression()
regressor.fit(data_train, target_train)

target_pred = regressor.predict(data_test)

predictions = pd.DataFrame(
    np.c_[target_test, target_pred],
    columns=['target_test', 'target_pred']
)
predictions['error'] = predictions['target_pred'] - predictions['target_test']
predictions['abs_error'] = abs(predictions['error'])
print(predictions.head(10))

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target']
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=42
)

regressor = GradientBoostingRegressor(random_state=42)
regressor.fit(data_train, target_train)
print(f'R^2 score: {regressor.score(data_test, target_test):.4f}')

import pickle

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target']
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=42
)

regressor = GradientBoostingRegressor()
regressor.fit(data_train, target_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

with open('model.pkl', 'rb') as file:
    regressor_loaded = pickle.load(file)

print(regressor_loaded)