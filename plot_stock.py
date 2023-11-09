import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import streamlit as st

register_matplotlib_converters()


# Function to plot the graph of stock prices of the company and save it in the folder
def plot_stock_prices(dataset, title):
    fig = plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Years")
    plt.ylabel("Close Price")
    plt.plot(dataset)
    st.pyplot(fig)

# Function to plot the graph of predicted and original stock prices of the company and save it in the folder
def plot_stock_prices_predicted(dataset, title, company_close_price, predictions):
    valid = dataset
    valid["Prediction_Price"] = predictions
    fig = plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Years")
    plt.ylabel("Close Price")
    plt.plot(company_close_price)
    plt.plot(valid[["Close", "Prediction_Price"]])
    plt.legend(["Original_train", "Valid_test", "Prediction_Price"])
    st.pyplot(fig)