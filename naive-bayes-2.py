import numpy as np
import pandas as pd

db = pd.read_csv("diabetes.csv")
print(db)
db.info()
db.describe()
db.shape
db.columns
db.dtypes
db.nunique()
db.isna().sum()
# plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(x="Age", y="DiabetesPedigreeFunction", data=db)
sns.regplot(x="Age", y="Pregnancies", data=db)
correlation_matrix = db.corr()
sns.heatmap(correlation_matrix, annot=True)
# assign
x = db.iloc[:, 0:-1].values
y = db.iloc[:, -1].values
print(x)
# split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# normalize
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# algorithm
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb = nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
print(y_test)
print(y_pred)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(nb.predict(sc.fit_transform([[1, 93, 70, 31, 0, 30.4, 0.315, 23]])))
