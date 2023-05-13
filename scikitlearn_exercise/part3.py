from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=20)