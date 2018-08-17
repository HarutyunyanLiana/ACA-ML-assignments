# The app has no user interface and can be improved a lot.
# The model does not perform so well and it is going to be improved for Homework 2.
# Example of URL to predict house price - /predict/full_sq=70/floor=3/life_sq=37/build_year=1990/max_floor=9


import os
from flask import Flask
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm, skew
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


train = pd.read_csv("train.csv", parse_dates=['timestamp'])
train = train.dropna(axis=1, how="all")
train = train[['full_sq', "floor", "life_sq", "build_year", "max_floor", "price_doc"]]
train = train.dropna()
numerical_features = train.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("price_doc")
train_num = train[numerical_features]
skewness = train_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_num[skewed_features] = np.log1p(train_num[skewed_features])

y = np.array(train.price_doc)
x_train, x_test, y_train, y_test = train_test_split(train_num, y, test_size=0.3, random_state=1)
stdscale = StandardScaler()
x_train.loc[:, numerical_features]=stdscale.fit_transform(x_train.loc[:, numerical_features])
x_test.loc[:, numerical_features]=stdscale.fit_transform(x_test.loc[:, numerical_features])

xgb_params = {
    'eta': 0.01,
    'max_depth': 10,
    'colsample_bytree': 1,
    'lambda': 0.5,
    'objective': 'regression',
    'metric': 'rmse',
    'silent': 0,
    'learning rate': 0.003
}
d_train = lgb.Dataset(x_train, y_train)
d_test = lgb.Dataset(x_test, y_test)

bst = lgb.train(xgb_params, d_train, 5000, valid_sets=[d_test], verbose_eval=50, early_stopping_rounds=100)
# y_pred = bst.predict(x_test)


app = Flask(__name__)


@app.route("/")
def index():
    return '<h1> Welcome to the humblest webpage about Russian House Dataset.</h1>'


@app.route('/predict/full_sq=<full_sq>/floor=<floor>/life_sq=<life_sq>/build_year=<build_year>/max_floor=<max_floor>')
def show_predict(full_sq, floor, life_sq, build_year, max_floor):
    d = {'full_sq': int(full_sq), 'floor': int(floor), 'life_sq': int(life_sq), "build_year": int(build_year), "max_floor": int(max_floor)}

    to_be_predicted = pd.DataFrame({k: [v] for k, v in d.items()})
    predicted = bst.predict(to_be_predicted)
    return "The predicted Price with that parameters will be <b>" + str(int(predicted[0])) + "</b"


if __name__ == "__main__":
    app.run()




