import numpy as np
import pandas as pd

sal = pd.read_csv("Salary_dataset.csv")
sal
sal.info()
sal.describe()
sal.shape
sal.dtypes
sal.nunique()
sal.isna().sum()
# 1. assign
x = sal[["YearsExperience"]].values
y = sal[["Salary"]].values
# 2. split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50)
# 3. Algorithm
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg = reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
y_test
from matplotlib import pyplot as plt

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, y_pred, color="red")
res = y_test - y_pred
plt.scatter(x_test, res, color="blue")
plt.plot(x_test, [0] * len(y_pred - 1), color="red")
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
