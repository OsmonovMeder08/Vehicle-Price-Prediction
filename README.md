# 🚗 Vehicle Price Prediction

Простой проект на **Machine Learning (Linear Regression)** для предсказания цены автомобиля.



## 📌 Описание

Этот проект использует модель линейной регрессии для предсказания стоимости автомобиля на основе следующих параметров:

* Year (год)
* Engine (объём двигателя)
* Cylinders (цилиндры)
* Mileage (пробег)
* Doors (двери)
* Make (марка)


## 🧠 Используемые технологии

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask (веб-интерфейс)


## 📊 Возможности

* Обучение модели (`train.py`)
* Предсказание через терминал (`predict.py`)
* Веб-приложение (`app.py`)
* Отображение метрики R²
* Таблица с данными
* Простая форма для ввода


## ⚙️ Установка (для новичков)

### 1. Клонировать проект

```bash
git clone https://github.com/OsmonovMeder08/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction
```


### 2. Создать виртуальное окружение

```bash
python -m venv .venv
```


### 3. Активировать его

#### Linux / Mac:

```bash
source .venv/bin/activate
```

#### Windows:

```bash
.venv\Scripts\activate
```


### 4. Установить зависимости

```bash
pip install pandas numpy scikit-learn flask
```


## 🚀 Запуск проекта

### 🔹 1. Обучить модель

```bash
python train.py
```

👉 создаст папку `models/` и файлы:

* model.pkl
* metrics.pkl


### 🔹 2. Запуск веб-приложения

```bash
python app.py
```

👉 открой в браузере:

```
http://127.0.0.1:5000
```


### 🔹 3. Предсказание через терминал

```bash
python predict.py
```


## 📈 Метрики

* R² Score отображается в веб-интерфейсе
* Чем ближе к **1.0**, тем лучше модель


## 📂 Структура проекта

```
Vehicle-Price-Prediction/
│
├── app.py
├── train.py
├── predict.py
├── Vehicle Price.csv
│
├── models/
│   ├── model.pkl
│   └── metrics.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── regression.png
```
