import pickle
import pandas as pd
import numpy as np

with open('models/model.pkl', 'rb') as f:
    pipeline, columns, make_te_map = pickle.load(f)

global_mean = np.mean(list(make_te_map.values()))

def preprocess(data):
    df = pd.DataFrame([data])

    df["car_age"] = 2026 - df["year"]
    df["log_mileage"] = np.log1p(df["mileage"])
    df["engine_per_cyl"] = df["engine"] / (df["cylinders"] + 1)
    df["engine_squared"] = df["engine"] ** 2
    df["make_te"] = df["make"].map(make_te_map).fillna(global_mean)

    return df.reindex(columns=columns, fill_value=0)

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Write only numbers!")

data = {
    "year": get_float("Year (for example 2020): "),
    "engine": get_float("Engine (for example 2): "),
    "cylinders": get_float("Cylinders (for example 4): "),
    "mileage": get_float("Mileage (for example 50000): "),
    "doors": get_float("Doors (for example 4): "),
    "make": input("Make (for example Toyota, BMW, Honda, Mercedes): ")
}

X = preprocess(data)
pred = np.exp(pipeline.predict(X)[0])


print("💰 Price:", round(pred, 2))