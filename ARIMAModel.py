import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import yfinance as yf

def Model(stockNo, TrainDateStart, TrainDateEnd, testStart, year, month, day):
    gs = yf.download(stockNo, start = TrainDateStart, end = TrainDateEnd)

    # Preprocess data
    dataset_ex_df = gs.copy()
    dataset_ex_df = dataset_ex_df.reset_index()
    dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
    dataset_ex_df.set_index('Date', inplace=True)
    dataset_ex_df = dataset_ex_df['Close'].to_frame()

    # Define the ARIMA model
    def arima_forecast(history):
        # Fit the model
        model = ARIMA(history, order=(0,1,0))
        model_fit = model.fit()
        
        # Make the prediction
        output = model_fit.forecast()
        yhat = output[0]
        return yhat

    # Split data into train and test sets
    X = dataset_ex_df.values
    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]

    # Walk-forward validation
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        # Generate a prediction
        yhat = arima_forecast(history)
        predictions.append(yhat)
        # Add the predicted value to the training set
        obs = test[t]
        history.append(obs)
    
    return predictions