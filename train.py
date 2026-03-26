import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import re

df = pd.read_csv("Vehicle Price.csv")
df = df[["make","year","engine","cylinders","mileage","doors","price"]].copy()

def extract_engine(x):
    try:
        x = str(x).lower()
        m = re.search(r'(\d+\.\d+)', x)
        if m:
            return float(m.group(1))
        m = re.search(r'\b([2-8])\b', x)
        if m:
            return float(m.group(1))
        return None
    except:
        return None

df["engine"] = df["engine"].apply(extract_engine)

for col in ["year","engine","cylinders","mileage","doors","price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
df = df[df["price"] > 1000]

df["car_age"] = 2024 - df["year"]
df["log_mileage"] = np.log1p(df["mileage"])
df["engine_per_cyl"] = df["engine"] / (df["cylinders"] + 1)
df["engine_squared"] = df["engine"] ** 2

y = np.log(df["price"])

make_mean = y.groupby(df["make"]).mean()
global_mean = y.mean()
df["make_te"] = df["make"].map(make_mean).fillna(global_mean)

X = df[[
    "year","engine","cylinders","mileage","doors",
    "car_age","log_mileage","engine_per_cyl","engine_squared","make_te"
]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R2:", r2)

y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump((pipeline, X.columns, make_mean.to_dict()), f)

with open("models/metrics.pkl", "wb") as f:
    pickle.dump({"r2": r2}, f)

plt.scatter(y_test_exp, y_pred_exp)

min_val = min(y_test_exp.min(), y_pred_exp.min())
max_val = max(y_test_exp.max(), y_pred_exp.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()

