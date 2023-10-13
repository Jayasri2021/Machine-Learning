import pandas as pd
import numpy as np

hb = pd.read_csv("headbrain.csv")
print(hb)
hb.info()
hb.describe()
hb.shape
hb.columns
hb.dtypes
hb.nunique()
hb.isna().sum()
# 1. ASSIGN
X = hb[["Head Size(cm^3)"]].values
y = hb[["Brain Weight(grams)"]].values
# 1. ASSIGN
X = hb[["Head Size(cm^3)"]].values
y = hb[["Brain Weight(grams)"]].values
# 2. SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Algorithm
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg = reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_test
from matplotlib import pyplot as plt

plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue")
res = y_test - y_pred
plt.scatter(X_test, res, color="red")
plt.plot(X_test, [0] * len(y_pred - 1), color="blue")
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
