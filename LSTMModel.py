import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd
#import pandas_datareader.data as web
from datetime import datetime, date, timedelta
import datetime as dt
import time
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os

def AdjustXaxis(date, day):
    add_day = timedelta(days = day-1)
    myday = dt.datetime.strptime(date, "%Y-%M-%S")
    return str(myday + add_day)

Epoch = 25
Batch_size = 32
Units = 50 ##
predictionDays = 90
layers = 1
dropout = 0.2
val = 0.2
predictModel = 'LSTM'

def Model(stockNo, TrainDateStart, TrainDateEnd, testStart, year, month, day):
    # Load Data
    df = yf.download(stockNo, start = TrainDateStart, end = TrainDateEnd)
    df = df.reset_index() 

    test_data = yf.download(stockNo, start = testStart)
    test_data = test_data.reset_index()

    # Prepare Data
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1)) 

    x_train = []
    y_train = []

    for i in range(predictionDays, len(scaled_data)):
        x_train.append(scaled_data[i-predictionDays : i, 0]) 
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train) 
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build The Model
    model = Sequential()
    model.add(LSTM(units = Units, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1)) 
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error', 'mean_absolute_error'])
    history = model.fit(x_train, y_train, epochs = Epoch, batch_size = Batch_size, validation_split = val)
    #print(model.summary())

    # Test The Model Accuracy on Existing Data
    # Load Test Data
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((df['Close'], test_data['Close']), axis = 0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictionDays:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Prediction on Test Data
    x_test = [] 

    for i in range(predictionDays, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays: i, 0])
        
    for i in range(len(model_inputs)-10, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays: i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices_1 = scaler.inverse_transform(predicted_prices)
    print("=======LSTM prediction=======")
    print(predicted_prices_1[len(predicted_prices_1)-8:])

    # Plot
    dates_actual = []
    dates_predict = []
    df_date_close = test_data['Date']
    daysToPredict = 10
    dayRange = 1
    #now = datetime.date()

    for i in range(len(df_date_close)):
        dates_actual.append(str(df_date_close[i].date()))
        dates_predict.append(str(df_date_close[i].date()))
        now = df_date_close[i].date()
    
    for i in range(daysToPredict):
        now = now + timedelta(days=dayRange)
        dates_predict.append(str(now))
    
    date_actual = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates_actual]    
    date_predict = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates_predict]

    return date_actual, date_predict, actual_prices, predicted_prices_1
