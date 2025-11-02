import streamlit as st



import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats
from streamlit_lottie import st_lottie
import requests
import feedparser
from urllib.parse import quote
# Example: Personalize welcome/assistant
if 'last_stocks' in st.session_state:
    st.info(f"Hi again! Last time you analyzed: {', '.join(st.session_state['last_stocks'])}")
# Then use st.session_state['last_metrics'] or ['last_years'] in your model/chart code!

def ai_answer(question):
    q = question.lower()
    if "mse" in q:
        return "MSE stands for Mean Squared Error. It's a measure of the difference between actual and predicted values (lower is better)."
    if "prophet" in q:
        return "Prophet is a time-series forecasting model from Meta that handles trends/seasonality for accurate predictions."
    if "how to use" in q or "guide" in q:
        return "Go to any tab, enter a stock symbol, select a date, and click the button. Results with charts and tables will appear."
    if "download" in q:
        return "Every chart or table in this app has a Download button (CSV for tables, PNG for charts) beneath it."
    if "team" in q or "contact" in q:
        return "Built by Sajid Basha (lead), Anish Kumar, Jeevan, Surya Prakash, VITS Data Science. Contact: team.vits.ds@gmail.com"
    # Add more Q/A here
    return "Sorry, I don't have an answer for that. Try another question!"


# ========== Sidebar title change ===============
st.sidebar.markdown("<h2 style='color:#2196f3;font-family:Poppins,sans-serif;font-weight:900;'>App Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Select a section below:")

# --- All the rest as before ---
try:
    import statsmodels.api as sm
except:
    sm = None

try:
    from prophet import Prophet
except:
    Prophet = None

def is_weekend(dt):
    return dt.weekday() >= 5

peer_dict = {
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "ORCL"],
    "GOOGL": ["AAPL", "MSFT", "META"],
    "TCS.NS": ["INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "RELIANCE.NS": ["BPCL.NS", "TATAMOTORS.NS", "ONGC.NS"],
    "INFY.NS": ["TCS.NS", "HCLTECH.NS", "TECHM.NS"],
}

branch_dict = {
    "AAPL": ["Cupertino HQ", "Austin Campus", "China Ops", "Apple Retail"],
    "RELIANCE.NS": ["Mumbai HQ", "Retail Branch 1", "Retail Branch 2", "Petrochemicals", "Jio Infocomm"],
    "TCS.NS": ["Hyderabad Campus", "Bangalore Campus", "UK Office", "US Subsidiary"],
    "MSFT": ["Redmond HQ", "Europe Division", "Asia Center"],
}

st.set_page_config(page_title="StellarStocks Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def rich_news_panel(ticker):
    query = quote(ticker)
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(rss_url)
        if feed.entries:
            st.markdown("""
            <div style="background: linear-gradient(94deg,#182131 65%,#193a4b 100%); border-radius:18px;
                        box-shadow: 0 0 12px #08e1be9c; padding:18px 30px 11px 26px; color:#efffff; margin-bottom:24px; margin-top:10px;">
                <span style="font-size:1.5rem;font-family:Orbitron,sans-serif;color:#30ffdd;font-weight:900;">📰 Market News</span>
            """, unsafe_allow_html=True)
            for entry in feed.entries[:5]:
                st.markdown(
                    f"<div style='margin-bottom:13px;'><b style='font-size:1.13rem;color:#27d396'>{entry.title}</b>"
                    f"<div style='font-size:0.96rem;color:#c0fffa'>{entry.published}</div>"
                    f"<a href='{entry.link}' target='_blank' style='display:inline-block;background:#28e17a;color:#143025;"
                    f"padding:4px 13px;font-weight:700;border-radius:6px;text-decoration:none;margin-top:3px;font-family:Poppins'>Read Full Article</a></div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No news found for this symbol.")
    except Exception as e:
        st.warning(f"Could not fetch news. ({e})")
        
st.markdown('''
    <style>
        .glow-card {
            background: #16191c;
            border-radius: 18px;
            border:2px solid #27ae60;
            box-shadow: 0 0 16px 4px #27ae6098;
            min-width: 340px;
            margin-bottom: 24px;
        }
        .glow-card.blue {
            border: 2px solid #2196f3;
            box-shadow: 0 0 16px 3px #2196f3a0;
        }
    </style>
''', unsafe_allow_html=True)

pages = [
    ("🏠 Home", "Home"),
    ("📊 Analysis", "Analysis"),
    ("🤖 Prediction", "Prediction"),
    ("📈 Advanced Visualization", "Viz"),
    ("👤 Portfolio Comparison", "Portfolio"),
    ("⏳ Distribution", "Distribution"),
    ("👥 Team", "Team"),
    ("ℹ️ About", "About"),
    ("✉️ Contact", "Contact")
]
st.sidebar.title("")
page = st.sidebar.radio("Go to", [p[0] for p in pages])
page_label_to_name = {p[0]: p[1] for p in pages}
page = page_label_to_name[page]

def download_figure(fig, fname):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label=f"Download {fname}.png",
        data=buf.getvalue(),
        file_name=f"{fname}.png",
        mime="image/png"
    )
    plt.close(fig)

def display_company_info(ticker, branch=None):
    info = None
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = None
    if info:
        name = info.get('longName', info.get('shortName', 'Unknown Company'))
        exchange = info.get('exchange', 'N/A')
        country = info.get('country', 'N/A')
        sector = info.get('sector', 'N/A')
        st.markdown(f"""
        <div class="glow-card" style="padding:20px 34px 12px 34px;">
        <h2 style="margin-bottom:11px; color:#27ae60">{name}</h2>
        <div style="display:flex;flex-wrap:wrap;gap:28px;font-size:1.13rem;">
          <div>
            <span style="color:#e7eaf2;">🏦 <b>Exchange</b></span><br>
            <span style="color:#b7d7ec;"><b>{exchange}</b></span>
          </div>
          <div>
            <span style="color:#e7eaf2;">🌎 <b>Country</b></span><br>
            <span style="color:#b7d7ec;"><b>{country}</b></span>
          </div>
          <div>
            <span style="color:#e7eaf2;">🏢 <b>Sector</b></span><br>
            <span style="color:#b7d7ec;"><b>{sector}</b></span>
          </div>
          {'<div><span style="color:#e7eaf2;">🏭 <b>Branch</b></span><br><span style="color:#b7d7ec;"><b>' + branch + '</b></span></div>' if branch else ''}
        </div>
        </div>
        """, unsafe_allow_html=True)

def branch_selector(ticker, key):
    branches = branch_dict.get(ticker.upper())
    if branches:
        return st.selectbox("Select Branch", branches, key=key)
    return None

def peer_selector(main_ticker):
    main_upper = main_ticker.upper()
    peers = peer_dict.get(main_upper, [])
    options = [p for p in peers if p != main_upper]
    if options:
        peer = st.selectbox("Select a peer/competitor/sector for comparison", options, key=f"peer-{main_ticker}")
        return peer
    return None

def get_data_safe(ticker, start, end):
    if not ticker:
        st.error("Please enter a stock symbol. (Tip: Use TCS.NS for Indian stocks, AAPL/TSLA for US stocks.)")
        return None
    if start > end:
        st.error("Start Date cannot be after End Date.")
        return None
    if end > date.today():
        st.warning("End Date is in the future; showing up to today's available data.")
        end = date.today()
    if is_weekend(start) or is_weekend(end):
        st.info("You've selected a weekend. Markets are closed; data may be missing.")
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.error("No data found. This may be because the symbol is incorrect, the company is delisted, or the dates are weekends/holidays.")
            return None
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Yahoo Finance/API error: {e}")
        return None

import plotly.express as px
def plot_heatmap(df, metric='Close'):
    heat_df = df.copy()
    heat_df['Month'] = heat_df['Date'].dt.to_period('M').astype(str)
    heat_df['Day'] = heat_df['Date'].dt.day
    pivot = heat_df.pivot_table(index='Month', columns='Day', values=metric, aggfunc='mean')
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale="Viridis", title=f"{metric} Heatmap")
    st.plotly_chart(fig, use_container_width=True)

import plotly.graph_objs as go
def plot_candlestick(df, title=""):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# HOME PAGE --------
if page == "Home":
    col1, col2 = st.columns([1,2])
    with col1:
        try:
            st.image("stockmarket_logo.png", width=120)
        except Exception:
            st.write("[Logo image missing]")
        lottie_json = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_0yfsb3a1.json")
        if lottie_json:
            st_lottie(lottie_json, height=180)
    with col2:
        st.markdown("<h1 style='color:#27ae60;margin-bottom:-10px;'>StellarStocks Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#b7d7ec;'>AI-powered Stock Market Insights · Tomorrow’s Trends Today</h4>", unsafe_allow_html=True)
        st.caption("Fast, interactive forecast & analysis dashboard. Try it free, powered by Python & ML.")
    st.markdown("---")
    st.markdown("#### App Highlights")
    feat1, feat2, feat3 = st.columns(3)
    feat1.markdown('<div class="glow-card blue" style="text-align:center;"><b>🔮 ML Models</b></div>', unsafe_allow_html=True)
    feat2.markdown('<div class="glow-card blue" style="text-align:center;"><b>📅 Date Search</b></div>', unsafe_allow_html=True)
    feat3.markdown('<div class="glow-card blue" style="text-align:center;"><b>📈 Download Tables/Graphics</b></div>', unsafe_allow_html=True)
    st.markdown("---")


# ANALYSIS PAGE --------
if page == "Analysis":
    st.header("📊 Stock Analysis")
    st.info("Tip: Enter the official stock symbol (e.g., AAPL for Apple (US), TCS.NS for TCS in India). Use a valid date range. Data may be missing on weekends or market holidays.")

    ticker = st.text_input("Stock Symbol", value="AAPL", help="Eg: AAPL for Apple (US), TCS.NS for Tata Consultancy (India).")
    start = st.date_input("Start Date", value=date.today() - timedelta(days=90), help="First trading day for data (must be before End Date).")
    end = st.date_input("End Date", value=date.today(), help="Last possible trading day for data (max: today).")

    st.markdown("_(For Indian stocks, use .NS, ex: TCS.NS, RELIANCE.NS.)_")
    st.caption("Click 'Fetch Data' to load company profile, news, table of prices, and chart.")

    if st.button("Fetch Data", help="Loads all available stock data for symbol and dates."):
        with st.spinner("Fetching company and price data…"):
            df = get_data_safe(ticker, start, end)

            if df is None:
                st.warning("No data was found for this selection. Please check if the symbol is correct (e.g. AAPL or TCS.NS), make sure markets were open on your selected dates, and try again.")
            elif df.empty:
                st.warning("Data returned empty—possibly due to a non-trading day, holiday, or wrong symbol. Try different dates or another company.")
            else:
                branch = branch_selector(ticker, "analysis-branch")
                display_company_info(ticker, branch)
                st.markdown(f'<h5 style="color:#27ae60;">Articles for {ticker}</h5>', unsafe_allow_html=True)
                rich_news_panel(ticker)

                table_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
                    columns={col: f"{ticker} {col}" for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
                )
                st.caption("Below table: All daily stock prices for your selection.")
                st.dataframe(table_data, use_container_width=True)

                csv = table_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Table as CSV", data=csv, file_name=f"{ticker}_analysis_table.csv", mime="text/csv")
                st.caption("Use the button above to export all table data as a CSV file.")

                fig1, ax1 = plt.subplots(figsize=(7, 3.9))
                ax1.plot(df['Date'], df['Close'], label=f"{ticker} Close", linewidth=2.2, color="#27ae60")
                ax1.set_title(f"{ticker} Closing Price")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Closing Price")
                ax1.legend()
                st.caption("Below: Closing Price vs Date for your selected stock.")
                st.pyplot(fig1)

                buf = io.BytesIO()
                fig1.savefig(buf, format="png")
                st.download_button(
                    label="Download Closing Price Chart (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{ticker}_closing_chart.png",
                    mime="image/png"
                )
                plt.close(fig1)
                plot_heatmap(df, 'Close')
if page == "Prediction":
    st.header("🤖 ML-based Prediction")
    st.info("Tip: Enter stock symbol & a date range with at least 31 trading days for model prediction. Models shown include classic regression, ARIMA, and Prophet.")

    ticker = st.text_input("Stock Symbol", value="AAPL", key="pred", help="Eg: AAPL, TCS.NS, RELIANCE.NS, etc.")
    start = st.date_input("Start Date (Prediction)", value=date.today()-timedelta(days=365), help="First day for prediction (go back further for better accuracy).")
    end = st.date_input("End Date (Prediction)", value=date.today(), help="Last day for prediction (max: today).")

    st.caption("Click 'Run Prediction' to see machine learning and time series model results for your stock.")

    if st.button("Run Prediction", help="Runs forecasting models (regression, ARIMA, Prophet) for the stock and dates."):
        with st.spinner("Running stock prediction models and preparing charts…"):
            df = get_data_safe(ticker, start, end)

            if df is None or df.empty:
                st.error("No data available for this stock or date range. Please check your symbol (try AAPL, TCS.NS), pick valid dates, and try again.")
            elif len(df) <= 30:
                st.error("Prediction models require at least 31 days of trading data. Please increase your date range.")
            else:
                branch = branch_selector(ticker, "prediction-branch")
                display_company_info(ticker, branch)
                features = df[['Open', 'High', 'Low', 'Volume']][:-1]
                targets = df['Close'][1:]
                split_idx = int(len(features) * 0.8)
                X_train, X_test = features[:split_idx], features[split_idx:]
                y_train, y_test = targets[:split_idx], targets[split_idx:]
                date_test = df['Date'].iloc[split_idx+1:]

                st.subheader("Classic Regression Models")
                st.caption("Below: Regression models trained on historical stock prices. MSE = Mean Squared Error, R2 = goodness of fit.")
                models = {
                    'Linear Regression': LinearRegression(),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'Random Forest': RandomForestRegressor(n_estimators=80, random_state=42)
                }
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write(f"{name} MSE: {mean_squared_error(y_test, y_pred):.4f} | R2: {r2_score(y_test, y_pred):.4f}")
                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    ax.plot(date_test, y_test, label="Actual", linewidth=2.2)
                    ax.plot(date_test, y_pred, label="Predicted", linestyle="--", linewidth=2)
                    ax.set_title(f"{name} | Predicted vs Actual Close")
                    ax.legend()
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Close Price")
                    st.caption(f"{name} - solid = actual, dashed = predicted.")
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    st.download_button(
                        label=f"Download {name} Prediction Chart (PNG)",
                        data=buf.getvalue(),
                        file_name=f"{ticker}_{name.replace(' ', '_').lower()}_prediction.png",
                        mime="image/png"
                    )
                    plt.close(fig)

                # ARIMA model
                if sm is not None:
                    st.subheader("ARIMA Forecast")
                    st.caption("ARIMA is an advanced time series model for stock price forecasting.")
                    close_series = df['Close']
                    tlen = int(len(close_series) * 0.8)
                    try:
                        model_arima = sm.tsa.ARIMA(close_series[:tlen], order=(5,1,0)).fit()
                        preds = model_arima.forecast(steps=len(close_series[tlen:]))
                        st.write(f"ARIMA MSE: {mean_squared_error(close_series[tlen:], preds):.4f}")
                        fig2, ax2 = plt.subplots(figsize=(7, 2.5))
                        ax2.plot(df['Date'], close_series, label="Actual")
                        ax2.plot(df['Date'][tlen:], preds, label="ARIMA Forecast", linestyle="--")
                        ax2.set_title("ARIMA Forecast vs Actual")
                        ax2.legend()
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel("Close Price")
                        st.caption("Orange = ARIMA-predicted trend. Blue = true past closing price.")
                        st.pyplot(fig2)
                        buf2 = io.BytesIO()
                        fig2.savefig(buf2, format="png")
                        st.download_button(
                            label="Download ARIMA Chart (PNG)",
                            data=buf2.getvalue(),
                            file_name=f"{ticker}_arima_prediction.png",
                            mime="image/png"
                        )
                        plt.close(fig2)
                    except Exception as e:
                        st.warning(f"ARIMA model could not run for this data: {e}")

                # Prophet model
                if Prophet is not None:
                    st.subheader("Prophet Forecast")
                    st.caption("Prophet is a robust ML tool for trend detection and future forecasting with confidence intervals.")
                    try:
                        temp = df.copy()
                        temp = temp.rename(columns={"Date": "ds", "Close": "y"})
                        temp = temp.loc[:, ['ds', 'y']]
                        temp['ds'] = pd.to_datetime(temp['ds'])
                        temp['y'] = pd.to_numeric(temp['y'], errors='coerce')
                        model = Prophet()
                        model.fit(temp)
                        period = 30
                        future = model.make_future_dataframe(periods=period)
                        forecast = model.predict(future)
                        st.write(f"Prophet {period}-day forward forecast (blue line: forecast, black dots: actual).")
                        fig3 = model.plot(forecast)
                        st.caption("Blue line = ML prediction, shaded = uncertainty, black dots = real data.")
                        st.pyplot(fig3)
                        buf3 = io.BytesIO()
                        fig3.savefig(buf3, format="png")
                        st.download_button(
                            label="Download Prophet Chart (PNG)",
                            data=buf3.getvalue(),
                            file_name=f"{ticker}_prophet_prediction.png",
                            mime="image/png"
                        )
                        plt.close(fig3)
                    except Exception as e:
                        st.warning(f"Prophet model could not run for this data: {e}")
                else:
                    st.info("Prophet not installed; pip install prophet.")
               

if page == "Viz":
    st.header("📈 Interactive Plotly Visualizations")
    ticker = st.text_input("Which ticker?", value="AAPL", key="viz")
    start = st.date_input("Visualize from", value=date.today()-timedelta(days=365))
    end = st.date_input("To", value=date.today())
    fetch = st.button("Fetch Data", key="vizfetch")

    if fetch:
        if not ticker or not start or not end:
            st.warning("Please enter ticker and dates before fetching data.")
        else:
            df = get_data_safe(ticker, start, end)
            

            if df is not None and not df.empty:
                branch = branch_selector(ticker, "viz-branch")
                display_company_info(ticker, branch)
                st.subheader("Candlestick Chart")
                try:
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Date']
                    # Check columns & nulls for safety
                    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().any().any():
                        st.warning("Data missing/incomplete for candlestick chart. Try another date range or symbol.")
                    else:
                        plot_candlestick(df, f"{ticker} Candlestick Chart")
                except Exception as e:
                    st.warning(f"Unable to display candlestick chart: {e}")
                st.subheader("Sector/Peer Heatmap")
                plot_heatmap(df, 'Close')
            else:
                st.info("No data available for this range—try different dates or symbol.")
    else:
        st.info("Set ticker and date, then click 'Fetch Data' to display visualizations.")

if page == "Portfolio":
    st.header("Portfolio/Peer Comparison Widget")
    tickers = st.text_area("Enter portfolio/peer tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    start = st.date_input("Start Date (Portfolio)", value=date.today()-timedelta(days=365))
    end = st.date_input("End Date", value=date.today())
    if st.button("Compare Portfolio"):
        ticklist = [t.strip() for t in tickers.split(',') if t.strip()]
        if not ticklist:
            st.warning("Please enter at least one valid stock symbol (e.g., AAPL, MSFT, TCS.NS).")
        else:
            result_df = pd.DataFrame()
            for t in ticklist:
                dft = get_data_safe(t, start, end)
                if dft is not None and not dft.empty:
                    dft['Ticker'] = t
                    result_df = pd.concat([result_df, dft])
            if result_df.empty:
                st.warning("No data was fetched for any of the entered stocks. Please check the tickers and date range.")
            else:
                pivot = result_df.pivot_table(index='Date', columns='Ticker', values='Close')
                pivot.columns = [str(c) for c in pivot.columns]
                pivot_reset = pivot.reset_index()
                pivot_reset.columns = [str(c) for c in pivot_reset.columns]
                fig = px.line(pivot_reset, x='Date', y=[col for col in pivot_reset.columns if col != "Date"], title="Portfolio Performance")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Export Comparison (CSV)", data=pivot.to_csv().encode(), file_name="portfolio_comparison.csv")



if page == "Distribution":
    st.header("⏳ Return Distribution & Stats")
    ticker = st.text_input("Stock Symbol", value="AAPL", key="dist")
    start = st.date_input("Start Date (Returns)", value=date.today()-timedelta(days=180))
    end = st.date_input("End Date", value=date.today())
    if st.button("Show Distribution"):
        df = get_data_safe(ticker, start, end)
        if df is None or df.empty:
            st.warning("No data was loaded for this period. Please verify your symbol and try again.")
        else:
            returns = df['Close'].pct_change().dropna()
            if returns.empty:
                st.info("No return data to display for this symbol and period.")
            else:
                mean = float(returns.mean())
                median = float(returns.median())
                std = float(returns.std())
                st.write(f"Mean: {mean:.6f}, Median: {median:.6f}, Std: {std:.6f}")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(returns, kde=True, stat="density", ax=ax, color="#52BE80")
                x = np.linspace(returns.min(), returns.max(), 100)
                ax.plot(x, scipy.stats.norm.pdf(x, mean, std), 'r-', lw=2, label="Normal Curve")
                ax.set_title(f'{ticker} Returns Distribution')
                ax.legend()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="Download Distribution Chart (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{ticker}_returns_distribution.png",
                    mime="image/png"
                )
                plt.close(fig)


if page == "Team":
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:22px;">
    """, unsafe_allow_html=True)

    imgs = ["anish.jpg", "sajid.jpg", "jeevan.jpg", "surya.jpg"]
    names = ["Anish Kumar", "Sajid Basha", "Jeevan", "Surya Prakash"]
    roles = ["Team Lead", "ML Engineer", "UI & Design", "QA/Testing"]
    desc = ["Project & ML Lead", "Backend & Modelling", "Frontend Visionary", "Debugging & Docs"]

    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            # Only wrap the image in try/except; always show card
            try:
                st.image(imgs[i], width=100, output_format='auto', caption=names[i])
            except Exception:
                pass  # Just skip the image if missing
            # Show the info card
            st.markdown(f"""
            <div style="background:linear-gradient(60deg,#202837,#28e17a44);border-radius:14px;padding:8px 2px;text-align:center;">
                <b style="font-size:1.08rem;color:#28e17a">{names[i]}</b><br>
                <span style="color:#a9edeb;font-weight:600">{roles[i]}</span><br>
                <span style="font-size:89%">{desc[i]}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



if page == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
<b>StellarStocks Predictor</b> is a professional-grade dashboard for stock analysis, market prediction, and peer/portfolio comparison.<br>
Developed by: <b>Sajid Basha (Lead), Anish Kumar, Jeevan, Surya Prakash</b><br>
<b>College:</b> Vignan Institute of Technology & Science (VITS), Department of Data Science, India<br>
<b>Guide:</b> Aparna<br>
<b>Tech Used:</b> Python, Streamlit, yfinance, scikit-learn, Prophet, statsmodels, plotly, seaborn, Yahoo Finance API, Lottie animation.<br>
Project completed: November 2025<br>
    """, unsafe_allow_html=True)

if page == "Contact":
    st.header("✉️ Contact / Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        msg = st.text_area("Your Feedback / Question")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you! Your feedback has been recorded.")

# --- Contact/project info ---
# --- Contact/project info ---
st.markdown("---")
st.markdown(
    "<div style='color:#b7d7ec; font-size:1.09rem; text-align:center;'>"
    "<b>StellarStocks</b> &copy; 2025 Team VITS Data Science &mdash; Developed for advanced, interactive stock analysis and forecasting with modern ML tools.<br>"
    "For academic, analytical, and professional demonstration purposes.<br>"
    "<span style='font-size:1rem'>Contact: <i>team.vits.ds@gmail.com</i></span>"
    "</div>",
    unsafe_allow_html=True
)

# --- AI PROJECT ASSISTANT IN SIDEBAR (ONLY ONCE!) ---

def ai_answer(question):
    q = question.lower()
    if "mse" in q:
        return "MSE stands for Mean Squared Error. It's a measure of the difference between actual and predicted values (lower is better)."
    if "prophet" in q:
        return "Prophet is a time-series forecasting model from Meta that handles trends/seasonality for accurate predictions."
    if "how to use" in q or "guide" in q:
        return "Go to any tab, enter a stock symbol, select a date, and click the button. Results with charts and tables will appear."
    if "download" in q:
        return "Every chart or table in this app has a Download button (CSV for tables, PNG for charts) beneath it."
    if "team" in q or "contact" in q:
        return "Built by Sajid Basha (lead), Anish Kumar, Jeevan, Surya Prakash, VITS Data Science. Contact: team.vits.ds@gmail.com"
    if "features" in q:
        return "App features: stock price charts, ML predictions, peer comparison, portfolio analysis, news, all exportable as CSV/PNG, and more."
    return "Sorry, I don't have an answer for that. Try another question!"

# Load and display the Lottie animation at the top of the sidebar (not inside expander)
lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_gnbhifng.json")
if lottie_json:
    st.sidebar.markdown("")  # Ensures following is in sidebar
    st_lottie(lottie_json, height=80, key="sidebar_bot")

with st.sidebar.expander("🤖 Assistant", expanded=False):
    st.write("**AI Project Assistant**\n\nType your question about the project or usage:")
    user_q = st.text_input("Type or choose your doubt:", key="assistant_query", placeholder="Eg: how to use, what is mse, guide...")
    if user_q:
        st.markdown(f"**Q:** {user_q}")
        st.success(ai_answer(user_q))
    st.markdown("Or select a common question for instant help:")
    choice = st.selectbox("Quick help", [
        "",
        "How to use the app?",
        "What is MSE?",
        "What is Prophet?", 
        "How to download results?",
        "Team/project info",
        "App features"
    ], key="assistant_choosedoubt")
    if choice and choice != "":
        st.markdown(f"**Q:** {choice}")
        st.success(ai_answer(choice))
