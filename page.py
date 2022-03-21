import streamlit as st
import os
import time
import yfinance as yf
import plotly.graph_objects as go
from PIL import Image, ImageOps
from datetime import date
import pandas as pd
import base64
from prophet import Prophet
from prophet.plot import plot_plotly


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

df = pd.read_csv("assets/documents/nasdaq_screener.csv")

with open('assets/documents/stock.txt', 'r') as file:
    stock_txt = file.read()
with open('assets/documents/rnn.txt', 'r') as file:
    rnn_txt = file.read()
with open('assets/documents/rnn2.txt', 'r') as file:
    rnn2_txt = file.read()
with open('assets/documents/rnn3.txt', 'r') as file:
    rnn3_txt = file.read()
with open('assets/documents/ann.txt', 'r') as file:
    ann_txt = file.read()
with open('assets/documents/ann2.txt', 'r') as file:
    ann2_txt = file.read()
with open('assets/documents/cnn.txt', 'r') as file:
    cnn_txt = file.read()
with open('assets/documents/tiingo.txt', 'r') as file:
    tiingo_txt = file.read()

file_ = open("assets/images/ann.gif", "rb")
contents = file_.read()
ann_gif = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("assets/images/cnn.gif", "rb")
contents = file_.read()
cnn_gif = base64.b64encode(contents).decode("utf-8")
file_.close()


def welcome():
    st.title("WELCOME TO STOCK BOT")
    st.write("- The goal of this app, is to help you visualize stock changes in real time and also predict how it'll "
             "move in the next few hours.")
    st.header("Why use Stock Prediction ?")
    st.write(stock_txt)
    st.header("How to predict the Stock Market ?")
    st.write(rnn_txt)
    st.header("LSTM Recurrent Neural Network")
    st.write(rnn2_txt)
    image = Image.open('assets/images/rnn.png')
    st.image(image, caption='RNN model Architecture', width=None)


def about():
    st.title("What are neural networks?")
    st.header("Artificial Neural Network (ANN)")
    st.write(ann_txt)
    st.markdown(
        f'<img src="data:image/gif;base64,{ann_gif}" alt="ann gif">',
        unsafe_allow_html=True,
    )
    st.write(ann2_txt)
    st.header("Convolution Neural Network (CNN)")
    st.write(cnn_txt)
    st.markdown(
        f'<img src="data:image/gif;base64,{cnn_gif}" alt="cnn gif">',
        unsafe_allow_html=True,
    )
    st.header("Recurrent Neural Network (RNN)")
    st.write("- Let us first try to understand the difference between an **RNN** and an **ANN** from the architecture "
             "perspective:")
    image = Image.open('assets/images/rnn2.png')
    st.image(image, caption='A looping constraint on the hidden layer of ANN turns to RNN.', width=None)
    st.write("**As you can see here, RNN has a recurrent connection on the hidden state. This looping constraint "
             "ensures that sequential information is captured in the input data.**")
    st.write(rnn3_txt)


def our_data():
    st.title("How we got OUR Data")
    st.write("In this project, we need to get the **historical** and **live stock** prices, and to do that we used "
             "multiple **API**'s to get **CSV** files.\n\nSome of the used **API**'s are :")
    st.header("Yahoo Finance")
    st.write("**yfinance** is not **affiliated**, **endorsed**, or **vetted** by **Yahoo, Inc**. It's an "
             "**open-source** tool that uses Yahoo's publicly available **API**s, and is intended for research and "
             "educational purposes.")
    with st.expander("Usage example of yfinance in Python:"):
        code = '''
        import pandas as pd
        import yfinance as yf
        df = yf.download('AAPL', 
                          start='2019-01-01', 
                          end='2021-06-12', 
                          progress=False,
        )
        df.head()
        '''
        st.code(code, language='python')
        st.write("Output:")
        image = Image.open('assets/images/yfinance.png')
        st.image(image, caption='Apple stock price', width=None)
        st.write("Data can also be saved into an **CSV** file")
        code = '''
        df.to_csv('AAPL.csv')
        '''
        st.code(code, language='python')
        st.write("Which creates a **CSV** file (excel) in your directory containing the whole dataframe")
    st.header("Tiingo")
    st.write(tiingo_txt)


def get_ticker(name):
    company = yf.Ticker(name)
    return company


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


def plot_raw_data(val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val.index, y=val['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=val.index, y=val['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def plot_go(data):
    # declare figure
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'], name='market data'))

    # Add titles
    fig.update_layout(
        title='live share price evolution',
        yaxis_title='Stock Price (USD per Shares)')

    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig)


def realtime():
    st.title("Data Viewer")
    with st.expander("View All Stock symbols and names:"):
        st.subheader("View of the stock")
        st.subheader("Stock names")
        st.table(df)
    choice = st.text_input("Stock Symbol", "")
    if choice:
        tick = get_ticker(choice)
        col1, col2, col3 = st.columns([1, 3, 1])
        col1.image(tick.info['logo_url'])
        col2.subheader(tick.info['shortName'])
        col3.download_button(
            label="Download data as CSV",
            data="",
            file_name='data.csv',
            mime='text/csv',
        )

        st.write(" # Info :")
        col1, col2, col3 = st.columns(3)
        current = tick.info['currentPrice']
        previous = tick.info['previousClose']
        change = round(current - previous, 2)
        currency = tick.info['financialCurrency']
        col1.metric("Current Price", str(current) + currency, change)
        col2.metric("Previous Close", str(previous) + currency, "0.00")
        col3.metric("Recommendations", tick.info['recommendationKey'].upper())
        # company = load_data(choice)
        val = yf.download(choice, START, TODAY)
        # st.write(tick.info)
        st.subheader("Today's data:")
        data = yf.download(choice, period="1d", interval="1m")
        st.write(data.tail())
        plot_raw_data(data)
        st.subheader("Historical data:")
        st.write(val.tail())
        plot_raw_data(val)
        plot_go(data)
        st.subheader("View live data for some stocks")
        st.button("Open View")


def predict_lstm():
    st.title("Stock prediction using LSTM")
    st.subheader("Getting the data")
    st.write("To get out **data** we use yfinance as follow: ")
    st.write("- First, we import our libraries")
    code = '''
    !pip install pandas
    !pip install yfinance
        
    import pandas as pd
    import yfinance as yf
        '''
    st.code(code, language="python")
    st.write("- After that, we download **historical data** about the stock wanted, and we save only the **closing "
             "prices** in a new dataframe as follow:")
    code = '''
    stock_symbol = 'GOOG'
    data = yf.download(tickers=stock_symbol,period='5y',interval='1d')
    close = data[['Close']]
            '''
    st.code(code, language="python")
    st.write("- If needed, we can plot the data as it is with :")
    code = '''
    close.plt()
            '''
    st.code(code, language="python")
    st.write("> Resulting in:")
    image = Image.open('assets/images/plot.png')
    st.image(image, caption='Plot historical data', width=None)
    st.write("- After we got out **data**, and saved the **closing price**, we normalize it between 0 "
             "& 1 with **MinMaxScaler** so that we can have **small values** that are easier for the program to "
             "understand")
    code = '''
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
            '''
    st.code(code, language="python")
    st.write("- Now that it is scaled properly, we can divide it to create our dataset")


def predict_prophet():
    st.title("Stock prediction using FbProphet")
    with st.expander("View All Stock symbols and names:"):
        st.subheader("View of the stock")
        st.subheader("Stock names")
        st.table(df)
    choice = st.text_input("Enter Stock symbol to start Prediction", "")
    if choice:
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(choice)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)



