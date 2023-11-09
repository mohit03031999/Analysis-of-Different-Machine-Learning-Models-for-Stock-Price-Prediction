import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics
import pandas as pd
import streamlit as st
import datetime
import plotly.express as px
import plotly.io as pio

import plot_stock

Path = "./"  # Path for reading the files

st.header('Stock Price Prediction using different Models')

# Defining the Test Dataset size
TEST_DATA_SIZE = 0.25

# Read the stocks input data
dataset = pd.read_csv(Path +'CAC40_stocks_2010_2021.csv')

# Creating a list of Company names from the dataset
dataset_company_list = list(dataset['CompanyName'].unique())
dataset_company_list.append('<Select>')
default_idx = dataset_company_list.index('<Select>')

# Company Name for which predictions needs to be done selected by the user
sidebar = st.sidebar
Select_company = sidebar.selectbox('Select the company from the below list',dataset_company_list,index=default_idx)
st.write('You selected:', Select_company)

starting_date = st.date_input("Select the start date",value=datetime.date(2010, 4, 1),min_value=datetime.date(2010, 4, 1))
ending_date = st.date_input("Select the end date",value=starting_date , min_value=starting_date,max_value=datetime.date(2021, 2, 1))

def date_difference(start,end):
    d1 = datetime.datetime.strptime(str(start), "%Y-%m-%d")
    d2 = datetime.datetime.strptime(str(end), "%Y-%m-%d")
    return abs((d2 - d1).days)

def stock_prediciton(USER_COMPANY,initial_date,end_date):

    # Defining the start date and end date of the stock prices for training the model
    START_DATE = initial_date.strftime("%Y-%m-%d")
    END_DATE = end_date.strftime("%Y-%m-%d")

    # Filter input dataset according to company name for which prediction needs to be done
    dataset_company = dataset[dataset['CompanyName'] == USER_COMPANY]
    dataset_company['Date'] = pd.to_datetime(dataset_company.Date,format='%Y-%m-%d')
    dataset_company.index = dataset_company['Date']
    dataset_Company_Symbol = str(dataset_company['StockName'].unique())

    # Filtering the dataset according to start date and end date and plot the dataset
    dataset_company = dataset_company[(dataset_company['Date'] > START_DATE) & (dataset_company['Date'] < END_DATE)]
    plot_stock.plot_stock_prices(dataset_company["Close"], dataset_Company_Symbol+"'s Stock Price")


    # In our case we are predicting the closing stock price of the company using opening stock price
    company_stock_price = dataset_company[["Close","Open"]]

    # Identify the length of the train and test data read the test size and split on sequence of train and test data.
    len_test_data = int(len(dataset_company) * TEST_DATA_SIZE)
    len_train_data = len(dataset_company) - len_test_data
    train_data = company_stock_price[:len_train_data]
    test_data = company_stock_price[len_train_data:]


    # LSTM
    # Firstly scaling all the prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_price = dataset_company.filter(["Close"])
    data_price.reset_index()
    scaled_price_data = scaler.fit_transform(data_price)

    # Splitting the scaled dataset into training data sets
    # For LSTM we are using last 70 values to predict the next value stock price
    training_price_data = scaled_price_data[0:len_train_data]
    x_train_lstm = []
    y_train_lstm = []
    for i in range(70, len(training_price_data)):
        x_train_lstm.append(training_price_data[i-70:i, 0])
        y_train_lstm.append(training_price_data[i, 0])

    # Converting into np array and reshaping the training dataset for LSTM Model
    x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
    x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0],x_train_lstm.shape[1],1))

    # Splitting the scaled dataset into testing dataset
    test_data = scaled_price_data[len_train_data - 70:]
    x_test_lstm = []
    y_test_lstm = dataset_company[len_train_data:]
    for i in range(70,len(test_data)):
        x_test_lstm.append(test_data[i-70:i,0])

    # Converting into np array and reshaping the training dataset for LSTM Model
    x_test_lstm = np.array(x_test_lstm)
    x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0],x_test_lstm.shape[1],1))

    # Build the LSTM model
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(units=60, return_sequences=True,input_shape=(x_train_lstm.shape[1],1)))
    LSTM_model.add(LSTM(units=60, return_sequences=False))
    LSTM_model.add(Dense(units=20))
    LSTM_model.add(Dense(units=1))

    LSTM_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM Model on training dataset
    LSTM_model.fit(x_train_lstm, y_train_lstm, batch_size=3, epochs=5)

    # Predicting values using LSTM and undo the scaling to get actual price value
    lstm_predict = LSTM_model.predict(x_test_lstm)
    lstm_price_predictions = scaler.inverse_transform(lstm_predict)

    # Plot the actual and predicted stock price
    plot_stock.plot_stock_prices_predicted(dataset_company[-len(x_test_lstm):], "LSTM Model for " + dataset_Company_Symbol+"'s Stock Price Prediction ",
                                             company_stock_price["Close"], lstm_price_predictions)


    # ARIMA,
    arima_training_data = company_stock_price['Close'][:len_train_data].values
    arima_testing_data = company_stock_price['Close'][len_train_data:].values

    past_values = [x for x in arima_training_data]
    arima_model_predict = []
    N_test = len(arima_testing_data)

    # ARIMA model parameters set as p=5, d=1, q=0
    for x in range(N_test):
        arima_model = ARIMA(past_values, order=(5, 1, 0))
        arima_fit = arima_model.fit(disp=0)
        output = arima_fit.forecast()
        y_cap = output[0]
        arima_model_predict.append(y_cap)
        actual_value = arima_testing_data[x]
        past_values.append(actual_value)

    plot_stock.plot_stock_prices_predicted(dataset_company[-len(arima_testing_data):], "ARIMA Model for " + dataset_Company_Symbol+"'s Stock Price Prediction ",
                                             company_stock_price["Close"], arima_model_predict)

    #Calculate RMSE values
    lstm_rmse = metrics.mean_squared_error(company_stock_price['Close'][len_train_data:], lstm_price_predictions, squared=False)
    arima_rmse = metrics.mean_squared_error(company_stock_price['Close'][len_train_data:], arima_model_predict, squared=False)

    st.write("RMSE for all the models: ")
    st.write("LSTM: ",lstm_rmse)
    st.write("ARIMA: ",arima_rmse)

# Run on streamlit if company selected by the user
if Select_company != '<Select>':
    if starting_date > ending_date:
        st.error('Error: End date must fall after start date.')
    elif date_difference(starting_date,ending_date) < 140:
        st.error('Error: Difference between two dates should be greater than 140.')
    else:
        stock_prediciton(Select_company,starting_date,ending_date)
