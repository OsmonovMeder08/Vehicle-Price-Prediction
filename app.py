from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

with open('models/model.pkl', 'rb') as f:
    pipeline, columns, make_te_map = pickle.load(f)

with open('models/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

r2 = metrics["r2"]
global_mean = np.mean(list(make_te_map.values()))

def preprocess(data):
    current_year = datetime.now().year
    
    df = pd.DataFrame([data])
    df["car_age"] = current_year - df["year"]
    df["log_mileage"] = np.log1p(df["mileage"])
    df["engine_per_cyl"] = df["engine"] / (df["cylinders"] + 1)
    df["engine_squared"] = df["engine"] ** 2
    df["make_te"] = df["make"].map(make_te_map).fillna(global_mean)

    return df.reindex(columns=columns, fill_value=0)

def create_plot():
    df = pd.read_csv("Vehicle Price.csv")

    def extract_engine(x):
        try:
            x = str(x).lower()
            m = re.search(r'(\d+\.\d+)', x)
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
    df["make_te"] = df["make"].map(make_mean).fillna(y.mean())

    X = df[columns]

    y_pred = pipeline.predict(X)

    y_test_exp = np.exp(y)
    y_pred_exp = np.exp(y_pred)

    plt.figure()
    plt.scatter(y_test_exp, y_pred_exp)

    min_val = min(y_test_exp.min(), y_pred_exp.min())
    max_val = max(y_test_exp.max(), y_pred_exp.max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode()

df = pd.read_csv("Vehicle Price.csv")

def extract_engine(x):
    try:
        x = str(x).lower()
        m = re.search(r'(\d+\.\d+)', x)
        if m:
            return float(m.group(1))
        return None
    except:
        return None

df["engine"] = df["engine"].apply(extract_engine)

for col in ["year","engine","cylinders","mileage","doors","price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.median(numeric_only=True))

df_display = df[["make","year","engine","cylinders","mileage","doors","price"]].copy()

df_display["engine"] = df_display["engine"].round().astype(int)
df_display["cylinders"] = df_display["cylinders"].astype(int)
df_display["mileage"] = df_display["mileage"].astype(int)
df_display["doors"] = df_display["doors"].astype(int)
df_display["price"] = df_display["price"].astype(int)

table_html = df_display.sample(10).to_html(index=False)

@app.route('/', methods=['GET','POST'])
def home():
    prediction = None
    plot_url = create_plot()

    if request.method == 'POST':
        data = {
            "year": float(request.form['year']),
            "engine": float(request.form['engine']),
            "cylinders": float(request.form['cylinders']),
            "mileage": float(request.form['mileage']),
            "doors": float(request.form['doors']),
            "make": request.form['make']
        }

        X = preprocess(data)
        pred = np.exp(pipeline.predict(X)[0])

        prediction = round(pred, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        r2=round(r2, 4),
        table=table_html,
        plot_url=plot_url
    )

if __name__ == "__main__":
    app.run(debug=True)