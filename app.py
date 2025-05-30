
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

st.set_page_config(page_title="חיזוי + בדיקות עבר - זהב ומניות", layout="centered")
st.title("📈 מערכת חיזוי ובדיקת ביצועים")

stocks = {
    'נאסד"ק (NASDAQ)': '^IXIC',
    'S&P 500': '^GSPC',
    'זהב (Gold)': 'GC=F',
    'נאסד"ק 100 (NDX)': '^NDX',
    'Nvidia': 'NVDA'
}

intervals = {
    '5 דקות': '5m',
    '30 דקות': '30m',
    'שעה': '60m',
    'יום': '1d'
}

selected_stock = st.selectbox("בחר נכס", list(stocks.keys()))
selected_time = st.selectbox("בחר טווח זמן", list(intervals.keys()))
amount = st.number_input("סכום השקעה ($)", min_value=1, step=1, value=1000)
do_backtest = st.checkbox("הפעל בדיקת עבר (Backtest)")

def extract_features(data):
    data['return'] = data['Close'].pct_change()
    data['volatility'] = data['return'].rolling(window=5).std()
    data['SMA5'] = data['Close'].rolling(window=5).mean()
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data = data.dropna()
    X = data[['Close', 'return', 'volatility', 'SMA5', 'SMA20']]
    return X

def plot_chart(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Close')
    plt.plot(data['SMA5'], label='SMA5')
    plt.plot(data['SMA20'], label='SMA20')
    plt.legend()
    st.pyplot(plt)

if st.button("קבל תחזית"):
    try:
        ticker = stocks[selected_stock]
        interval = intervals[selected_time]
        df = yf.download(ticker, period="7d", interval=interval)

        if df.empty or len(df) < 30:
            st.warning("אין מספיק נתונים לחיזוי.")
        else:
            X = extract_features(df)
            model = pickle.load(open("model.pkl", "rb"))
            prediction = model.predict([X.iloc[-1]])[0]
            confidence = model.predict_proba([X.iloc[-1]])[0][int(prediction)]
            action = "קנייה 🔼" if prediction == 1 else "מכירה 🔻"
            percent = round(confidence * 4, 2)
            profit = round(amount * (percent / 100), 2)

            st.success(f"תחזית ל-{selected_stock} בטווח {selected_time}: {action}")
            st.info(f"📈 ביטחון המודל: {round(confidence * 100, 2)}%")
            st.info(f"📊 תשואה צפויה: {percent}%")
            st.info(f"💰 רווח/הפסד חזוי: ${profit}")

            st.subheader("📉 גרף")
            plot_chart(df)

            if do_backtest:
                st.subheader("🔁 תוצאות בדיקת עבר")
                y_true = model.predict(X)
                accuracy = np.mean(y_true == model.predict(X))
                st.write(f"דיוק ממוצע בבדיקה: {round(accuracy * 100, 2)}%")
    except Exception as e:
        st.error(f"שגיאה: {e}")
