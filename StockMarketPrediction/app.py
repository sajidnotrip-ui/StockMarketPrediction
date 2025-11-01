import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.title("Stock Market Prediction App")
stock_symbol = st.text_input("Enter Stock Symbol", value="AAPL")

if stock_symbol:
    df = yf.download(stock_symbol, period="1y", interval="1d")
    st.write("Data preview:", df.head())

    if not df.empty:
        close = df['Close'].iloc[:,0] if isinstance(df['Close'], pd.DataFrame) else df['Close']

        st.subheader("Statistics")
        st.write(df.describe())

        st.write(f"Mean Closing Price: {close.mean():.2f}")
        st.write(f"Median Closing Price: {close.median():.2f}")
        st.write(f"Variance: {close.var():.2f}")
        st.write(f"Standard Deviation: {close.std():.2f}")

        # Price Trend Chart
        st.subheader("Closing Price Trend")
        fig, ax = plt.subplots()
        ax.plot(close)
        st.pyplot(fig)

        # Moving Averages Chart
        ma_10 = close.rolling(window=10).mean()
        ma_50 = close.rolling(window=50).mean()
        fig_ma, ax_ma = plt.subplots()
        ax_ma.plot(close, label="Close")
        ax_ma.plot(ma_10, label="10-Day MA", linestyle='--')
        ax_ma.plot(ma_50, label="50-Day MA", linestyle=':')
        ax_ma.legend()
        st.subheader("Moving Averages")
        st.pyplot(fig_ma)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr = df.corr()
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # ML: Multiple Regression
        st.subheader("ML Predictions Comparison")
        features = df[['Open', 'High', 'Low', 'Volume']][:-1]
        targets = close[1:]
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**{name}: MSE={mse:.2f}, R²={r2:.4f}**")
            fig_mdl, ax_mdl = plt.subplots()
            ax_mdl.plot(np.arange(len(y_test)), y_test, label="Actual")
            ax_mdl.plot(np.arange(len(y_pred)), y_pred, label="Predicted", linestyle='--')
            ax_mdl.set_title(f"{name} Actual vs Predicted")
            ax_mdl.legend()
            st.pyplot(fig_mdl)

        st.success("All results and charts displayed above!")
