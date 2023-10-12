import pandas as pd

hb = pd.read_csv("headbrain.csv")
print(hb)
hb.info()
hb.describe
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
