#  CRYPTO DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from textblob import TextBlob
from tensorflow.keras.models import load_model

#  PATHS
DATA_PATH = r"your Cleaned_bitcoin_2014_2025.csv"
ARIMA_PATH = r"arima_model.pkl"
LSTM_PATH = r"lstm_model.h5"
PROPHET_PATH = r"prophet_model.pkl"
SCALER_PATH = r"scaler_model.pkl"

# CONFIG
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title(" Crypto Intelligence Dashboard")


#  LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df['Close'] = df['Close'].replace(r'[\$,]', '', regex=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.fillna(method='ffill', inplace=True)

    return df

df = load_data()

# ------------------------------
#  FEATURES
df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(30).std()
df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()
df['signal'] = np.where(df['returns'] > 0, "BUY", "SELL")

# ------------------------------
#  SENTIMENT
news = [
    "Bitcoin booming",
    "Market crash fears",
    "Investors optimistic"
]
news_df = pd.DataFrame(news, columns=['text'])
news_df['sentiment'] = news_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# ------------------------------
#  NAVIGATION
page = st.sidebar.radio(" Navigate", [
    "Overview",
    "Price Explorer",
    "Forecast",
    "Sentiment",
    "Volatility",
    "Indicators",
    "Correlation",
    "Market Structure",
    "Model Performance",
    "Backtest",
    "Explorer"
])

#  OVERVIEW  And To Visualizing the Below 
if page == "Overview":

    st.subheader(" Executive KPIs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", round(df['Close'].iloc[-1],2))
    col2.metric("Volatility", round(df['volatility'].iloc[-1],5))
    col3.metric("Sentiment", round(news_df['sentiment'].mean(),2))

    st.plotly_chart(px.line(df, y='Close'), use_container_width=True)


#  PRICE EXPLORER
elif page == "Price Explorer":

    st.subheader(" Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Close'],
        high=df['Close'],
        low=df['Close'],
        close=df['Close']
    )])

    st.plotly_chart(fig, use_container_width=True)

# FORECAST
elif page == "Forecast":

    st.subheader(" Forecast Models")

    model = st.selectbox("Select Model", ["ARIMA", "LSTM", "Prophet"])

    if model == "ARIMA":
        model = joblib.load(ARIMA_PATH)
        forecast = model.forecast(30)
        st.line_chart(forecast)

    elif model == "LSTM":
        lstm_model = load_model(LSTM_PATH, compile=False)  
        scaler = joblib.load(SCALER_PATH)

        data = scaler.transform(df['Close'].values.reshape(-1,1))
        X = data[-60:].reshape(1,60,1)

        pred = lstm_model.predict(X)
        pred = scaler.inverse_transform(pred)

        st.success(f"Next Price: {pred[0][0]:.2f}")

    elif model == "Prophet":
        model = joblib.load(PROPHET_PATH)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        st.line_chart(forecast['yhat'])


#  SENTIMENT
elif page == "Sentiment":

    st.subheader("Sentiment Analysis")
    st.plotly_chart(px.bar(news_df, y='sentiment'))

    st.subheader("Distribution")
    st.plotly_chart(px.bar(news_df, y='sentiment', color='sentiment'))

#  VOLATILITY
elif page == "Volatility":

    st.subheader(" Volatility")
    st.plotly_chart(px.line(df, y='volatility'))

#  INDICATORS
elif page == "Indicators":

    st.subheader(" Moving Averages")
    st.plotly_chart(px.line(df[['Close','MA50','MA200']]))

#  CORRELATION
elif page == "Correlation":

    st.subheader(" Correlation Matrix")
    st.dataframe(df[['Close','returns','volatility']].corr())

#  BACKTEST
elif page == "Backtest":

    st.subheader(" Strategy Backtest")

    df['signal'] = np.where(df['returns']>0,1,-1)
    df['strategy'] = df['signal'].shift(1) * df['returns']

    st.line_chart((1+df['strategy']).cumprod())

#  EXPLORER
elif page == "Explorer":

    st.subheader(" Data Explorer")

    st.dataframe(df.tail())

    csv = df.to_csv().encode()
    st.download_button("Download CSV", csv, "crypto_data.csv")

### Model Performance
elif page == "Model Performance":

    st.subheader(" Model Performance Comparison")

    # Dummy RMSE (replace with real if available)
    performance = pd.DataFrame({
        "Model": ["ARIMA", "LSTM", "Prophet"],
        "RMSE": [1200, 900, 1000]
    })

    # Bar Chart
    st.plotly_chart(px.bar(performance, x="Model", y="RMSE", color="Model"))

    # Scatter Plot (optional comparison)
    st.plotly_chart(px.scatter(performance, x="Model", y="RMSE", size="RMSE"))

    st.write("Lower RMSE = Better Model")

### MARKET STRUCTURE
elif page == "Market Structure":

    st.subheader(" Market Structure Analysis")

    # Trend classification
    df['trend'] = np.where(df['Close'] > df['MA50'], "Bull", "Bear")

    # Donut Chart
    trend_counts = df['trend'].value_counts()

    fig = px.pie(
        values=trend_counts.values,
        names=trend_counts.index,
        hole=0.5,
        title="Market Regime Distribution"
    )

    st.plotly_chart(fig)

    # Scatter Plot
    st.subheader("Returns vs Volatility")

    scatter = px.scatter(
        df,
        x='returns',
        y='volatility',
        color='trend',
        title="Market Risk Structure"
    )

    st.plotly_chart(scatter)

#### DONUT CHART --- BUY/SELL
signal_counts = df['signal'].value_counts()

fig = px.pie(
    values=signal_counts.values,
    names=signal_counts.index,
    hole=0.5,
    title="Buy vs Sell Distribution"
)

st.plotly_chart(fig)

####  SCATTER PLOT --- GLOBAL
st.subheader(" Returns vs Volatility Scatter")

fig = px.scatter(
    df,
    x='returns',
    y='volatility',
    color='Close',
    title="Risk vs Return"
)

st.plotly_chart(fig)

##### BARPLOT ----  SENTIMENT DISTRIBUTION
st.subheader(" Sentiment Distribution")

fig = px.bar(news_df, y='sentiment', color='sentiment')
st.plotly_chart(fig)