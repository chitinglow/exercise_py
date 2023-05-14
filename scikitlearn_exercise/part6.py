import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)

clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f'Accuracy: {acc:.4f}')

classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [6, 7, 8],
    'min_samples_leaf': [4, 5]
}

import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)

classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [6, 7, 8],
    'min_samples_leaf': [4, 5]
}

grid_search = GridSearchCV(
    classifier,
    param_grid=param_grid,
    n_jobs=-1,
    scoring='accuracy',
    cv=2
)
grid_search.fit(X_train, y_train)
grid_search.score(X_test, y_test)
print(grid_search.best_params_)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python'
]


vectorizer = CountVectorizer()

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())
print(df)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python'
]

vectorizer = CountVectorizer(stop_words='english')

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())
print(df)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())
print(df)

import numpy as np
from numpy.linalg import norm
import pandas as pd
import random

np.random.seed(42)
df = pd.read_csv('data.csv')

x1_min = df.x1.min()
x1_max = df.x1.max()

x2_min = df.x2.min()
x2_max = df.x2.max()

centroid_1 = np.array([
    random.uniform(x1_min, x1_max),
    random.uniform(x2_min, x2_max)
])
centroid_2 = np.array([
    random.uniform(x1_min, x1_max),
    random.uniform(x2_min, x2_max)
])

data = df.values

for i in range(10):
    clusters = []
    for point in data:
        centroid_1_dist = norm(centroid_1 - point)
        centroid_2_dist = norm(centroid_2 - point)
        cluster = 1
        if centroid_1_dist > centroid_2_dist:
            cluster = 2
        clusters.append(cluster)

    df['cluster'] = clusters

    centroid_1 = [
        round(df[df.cluster == 1].x1.mean(), 3),
        round(df[df.cluster == 1].x2.mean(), 3)
    ]
    centroid_2 = [
        round(df[df.cluster == 2].x1.mean(), 3),
        round(df[df.cluster == 2].x2.mean(), 3)
    ]

print(centroid_1)
print(centroid_2)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

np.random.seed(42)
df = pd.read_csv('clusters.csv')

kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
kmeans.fit(df)

print(kmeans.cluster_centers_)