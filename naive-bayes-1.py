import numpy as np
import pandas as pd

ads = pd.read_csv("Social_Network_Ads.csv")
ads
ads.info()
ads.describe()
ads.shape
ads.columns
ads.dtypes
ads.nunique()
ads.isna().sum()
# assign
x = ads.iloc[:, 1:-1].values
y = ads.iloc[:, -1].values
x
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
# split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Normalize
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# algorithm
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb = nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
y_test
y_pred
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(nb.predict(sc.fit_transform([[1, 36, 33000]])))
