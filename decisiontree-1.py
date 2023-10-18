import numpy as np
import pandas as pd

db = pd.read_csv("Diabetes-2.csv")
db.info()
db.describe()
db.shape
db.columns
db.dtypes
db.nunique()
db.isna().sum()
# Assign
x = db.iloc[:, :-1].values
y = db.iloc[:, -1].values
print(x)
print(y)
# Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Normalize
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# Algorithm
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier()
dc = dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
dc1 = DecisionTreeClassifier(criterion="entropy", max_depth=6)
dc1 = dc1.fit(x_train, y_train)
y_pred1 = dc1.predict(x_test)
print(accuracy_score(y_test, y_pred1))
