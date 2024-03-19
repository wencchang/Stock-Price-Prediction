import numpy as np
from numpy import absolute
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import date, datetime, timedelta
import datetime as dt
import time
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
import os

def AdjustXaxis(start, day):
    add_day = timedelta(days = day-1)
    myday = dt.datetime.strptime(start, "%Y-%M-%S")
    return str(myday + add_day)

n_estimators = 100
predictionDays = 90
learning_rate = 0.05
early_stopping_rounds = 5
predictModel = 'XGBoost'
end_time_of_train = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')

def Model(stockNo, TrainDateStart, TrainDateEnd, testStart, year, month, day):
    from datetime import date
    testStart_date = date(year, month, day)
    todayDate = date.today()
    daysFuture = todayDate - testStart_date
    future = AdjustXaxis(testStart, daysFuture.days)

    # Load Data
    df = yf.download(stockNo, start = TrainDateStart, end = TrainDateEnd)
    df = df.reset_index() 
    print(df) 

    test_data = yf.download(stockNo, start = testStart)
    test_data = test_data.reset_index()

    # Prepare Data
    scaled_data = df['Close'].values.reshape(-1, 1)

    x_train = []
    y_train = []

    for i in range(predictionDays, len(scaled_data)):
        x_train.append(scaled_data[i-predictionDays : i, 0]) 
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train) 

    # Test The Model Accuracy on Existing Data
    # Load Test Data
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((df['Close'], test_data['Close']), axis = 0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictionDays:].values
    model_inputs = model_inputs.reshape(-1, 1)

    # Make Prediction on Test Data
    x_test = [] 
    #y_test = []
    for i in range(predictionDays, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays:i, 0])
        #y_test.append(model_inputs[i, 0])

    for i in range(len(model_inputs)-10, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays: i, 0])

    # Build The Model
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate = learning_rate, early_stopping_rounds=early_stopping_rounds)
    model.fit(x_train, y_train, eval_metric=mean_squared_error)

    x_test = np.array(x_test)
    predicted_prices = model.predict(x_test)
    #MSE = str(mean_squared_error(predicted_prices, y_test))
    #print("XGBoost Mean Squared Error : " + MSE)

    print("=======XGBoost prediction=======")
    print(predicted_prices[len(predicted_prices)-8:])

    # Plot
    dates_actual = []
    dates_predict = []
    df_date_close = test_data['Date']
    daysToPredict = 10
    dayRange = 1

    for i in range(len(df_date_close)):
        dates_actual.append(str(df_date_close[i].date()))
        dates_predict.append(str(df_date_close[i].date()))
        now = df_date_close[i].date()

    for i in range(daysToPredict):
        now = now + timedelta(days=dayRange)
        dates_predict.append(str(now))

    date_actual = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates_actual]
    date_predict = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates_predict]

    return date_actual, date_predict, actual_prices, predicted_prices
