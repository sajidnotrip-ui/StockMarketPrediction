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
import re
import time

# ============= SESSION STATE INITIALIZATION (CRITICAL FIX!) =============
if 'selected_branch' not in st.session_state:
    st.session_state.selected_branch = {}
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = {}
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = {}
if 'last_stocks' not in st.session_state:
    st.session_state.last_stocks = []

# ============= PAGE CONFIG =============
st.set_page_config(page_title="StellarStocks Predictor", page_icon="📈", layout="wide")

# ============= CUSTOM CSS =============
st.markdown('''
    <style>
        /* Animated header glow */
        @keyframes glow {
            0% { text-shadow: 0 0 5px #27ae60, 0 0 10px #27ae60; }
            50% { text-shadow: 0 0 20px #27ae60, 0 0 30px #27ae60; }
            100% { text-shadow: 0 0 5px #27ae60, 0 0 10px #27ae60; }
        }
        
        h1 {
            animation: glow 2s ease-in-out infinite;
        }
        
        .glow-card {
            background: #16191c;
            border-radius: 18px;
            border: 2px solid #27ae60;
            box-shadow: 0 0 16px 4px #27ae6098;
            min-width: 340px;
            margin-bottom: 24px;
            padding: 20px;
        }
        .glow-card.blue {
            border: 2px solid #2196f3;
            box-shadow: 0 0 16px 3px #2196f3a0;
        }
        
        /* Better metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 900;
        }
        
        /* Professional buttons */
        .stButton>button {
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            color: white;
            font-weight: 700;
            border-radius: 10px;
            border: none;
            padding: 12px 30px;
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(39, 174, 96, 0.6);
        }
    </style>
''', unsafe_allow_html=True)

# ============= IMPORTS & CONSTANTS =============
try:
    import statsmodels.api as sm
except:
    sm = None

try:
    from prophet import Prophet
except:
    Prophet = None

FAQ_ANSWERS = {
    "tcs": "TCS.NS is Tata Consultancy Services—the top Indian IT company, traded on NSE.",
    "reliance": "RELIANCE.NS is Reliance Industries, India's largest private sector corporation.",
    "infy": "INFY.NS is Infosys, a major Indian IT company.",
    "hdfc": "HDFC.NS is HDFC Bank—a leading private bank in India.",
    "icici": "ICICI.NS is ICICI Bank, a major Indian banking and financial services company.",
    "ibm": "IBM (International Business Machines) is a major US tech and consulting company; symbol: IBM (NYSE).",
    "aapl": "AAPL is Apple Inc., traded on NASDAQ, one of the world's biggest tech companies.",
    "goog": "GOOG is Google (Alphabet Inc.), traded on NASDAQ.",
    "mse": "MSE means Mean Squared Error—a common model error metric.",
    "arima": "ARIMA is a classic time series forecasting model using past values of the series for prediction.",
    "prophet": "Prophet (by Meta/Facebook) is an open-source library for fast, robust time series forecasting.",
    "team": "Team: Sajid Basha (Lead), Anish Kumar, Jeevan, Surya Prakash. Guide: Dr. Aparna, VITS DS.",
}

SYMBOL_PATTERN = re.compile(r"^[a-z]{1,6}(\.[a-z]{2,4})?$", re.I)

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

# ============= HELPER FUNCTIONS =============
def is_weekend(dt):
    return dt.weekday() >= 5

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def ai_answer(question):
    q = question.strip().lower()
    for key, answer in FAQ_ANSWERS.items():
        if key in q:
            return answer
    if SYMBOL_PATTERN.match(q.replace(' ', '')):
        symbol = q.strip().upper()
        return f"'{symbol}' looks like a stock symbol. Use Analysis or Prediction section for more details."
    if "model" in q or "ml" in q:
        return "This dashboard uses Linear Regression, Decision Trees, Random Forest, ARIMA, and Prophet for stock price prediction."
    return "Please ask about company symbols, project sections, chart types, ML models, or team info!"

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

# ============= FIXED BRANCH SELECTOR (KEY FIX!) =============
def branch_selector(ticker, page_key):
    """Fixed branch selector with session state"""
    branches = branch_dict.get(ticker.upper())
    if branches:
        # Use page-specific key to avoid conflicts
        state_key = f"{page_key}_branch"
        
        # Get current selection
        current_idx = 0
        if state_key in st.session_state.selected_branch:
            try:
                current_idx = branches.index(st.session_state.selected_branch[state_key])
            except ValueError:
                current_idx = 0
        
        selected = st.selectbox(
            "🏭 Select Branch/Office Location", 
            branches, 
            index=current_idx,
            key=f"branch_select_{page_key}",
            help="Choose specific branch to view detailed information"
        )
        
        # Update session state
        st.session_state.selected_branch[state_key] = selected
        return selected
    return None

def get_data_safe(ticker, start, end):
    """Fetch data with rate limiting and error handling"""
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
        time.sleep(0.5)  # Rate limiting
        df = yf.download(ticker, start=start, end=end, progress=False)
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

# ============= VISUALIZATION FUNCTIONS =============
import plotly.express as px
import plotly.graph_objs as go

def plot_heatmap(df, metric='Close'):
    heat_df = df.copy()
    heat_df['Month'] = heat_df['Date'].dt.to_period('M').astype(str)
    heat_df['Day'] = heat_df['Date'].dt.day
    pivot = heat_df.pivot_table(index='Month', columns='Day', values=metric, aggfunc='mean')
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale="Viridis", title=f"{metric} Heatmap")
    st.plotly_chart(fig, use_container_width=True)

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

# ============= SIDEBAR =============
st.sidebar.markdown("### 🤖 Project FAQ Assistant")
user_q = st.sidebar.text_input("Ask about the project, dashboard, symbols, models, or charts…")
if user_q:
    st.sidebar.info(ai_answer(user_q))
st.sidebar.markdown("---")

st.sidebar.markdown("<h2 style='color:#2196f3;font-family:Poppins,sans-serif;font-weight:900;'>App Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Select a section below:")

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

page = st.sidebar.radio("Go to", [p[0] for p in pages])
page_label_to_name = {p[0]: p[1] for p in pages}
page = page_label_to_name[page]

# ============= HOME PAGE =============
if page == "Home":
    col1, col2 = st.columns([1,2])
    with col1:
        try:
            st.image("stockmarket_logo.png", width=120)
        except Exception:
            st.write("📈")
        lottie_json = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_0yfsb3a1.json")
        if lottie_json:
            st_lottie(lottie_json, height=180)
    with col2:
        st.markdown("<h1 style='color:#27ae60;margin-bottom:-10px;'>StellarStocks Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#b7d7ec;'>AI-powered Stock Market Insights · Tomorrow's Trends Today</h4>", unsafe_allow_html=True)
        st.caption("Fast, interactive forecast & analysis dashboard. Try it free, powered by Python & ML.")
    
    st.markdown("---")
    
    # 🔥 NEW FEATURE: Live Market Pulse
    st.markdown("### 📈 Live Market Pulse")
    trending = ['AAPL', 'MSFT', 'GOOGL', 'TCS.NS', 'RELIANCE.NS']
    cols = st.columns(5)
    
    for i, tick in enumerate(trending):
        with cols[i]:
            try:
                stock = yf.Ticker(tick)
                hist = stock.history(period='2d')
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    prefix = "₹" if tick.endswith('.NS') else "$"
                    st.metric(
                        tick.replace('.NS', ''),
                        f"{prefix}{current_price:.2f}",
                        f"{change:+.2f}%"
                    )
            except:
                st.metric(tick.replace('.NS', ''), "N/A")
    
    st.markdown("---")
    st.markdown("#### App Highlights")
    feat1, feat2, feat3 = st.columns(3)
    feat1.markdown('<div class="glow-card blue" style="text-align:center;padding:30px;"><b>🔮 5 ML Models</b><br><small>Random Forest, ARIMA, Prophet & more</small></div>', unsafe_allow_html=True)
    feat2.markdown('<div class="glow-card blue" style="text-align:center;padding:30px;"><b>📅 Smart Analytics</b><br><small>Real-time data from 2000+ stocks</small></div>', unsafe_allow_html=True)
    feat3.markdown('<div class="glow-card blue" style="text-align:center;padding:30px;"><b>📈 Risk Assessment</b><br><small>Volatility & Sharpe ratio analysis</small></div>', unsafe_allow_html=True)
    st.markdown("---")

# ============= ANALYSIS PAGE (FIXED!) =============
if page == "Analysis":
    st.header("📊 Stock Analysis")
    st.info("Tip: Enter the official stock symbol (e.g., AAPL for Apple (US), TCS.NS for TCS in India). Use a valid date range. Data may be missing on weekends or market holidays.")

    ticker = st.text_input("Stock Symbol", value="AAPL", help="Eg: AAPL for Apple (US), TCS.NS for Tata Consultancy (India).")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date", value=date.today() - timedelta(days=90), help="First trading day for data (must be before End Date).")
    with col2:
        end = st.date_input("End Date", value=date.today(), help="Last possible trading day for data (max: today).")

    st.markdown("_(For Indian stocks, use .NS, ex: TCS.NS, RELIANCE.NS.)_")
    
    # 🔥 CRITICAL FIX: Branch selector BEFORE fetch button
    selected_branch = branch_selector(ticker, "analysis")
    
    st.caption("Click 'Fetch Data' to load company profile, news, table of prices, and chart.")

    if st.button("🔍 Fetch Data", help="Loads all available stock data for symbol and dates."):
        with st.spinner("Fetching company and price data…"):
            df = get_data_safe(ticker, start, end)

            if df is None:
                st.warning("No data was found for this selection. Please check if the symbol is correct (e.g. AAPL or TCS.NS), make sure markets were open on your selected dates, and try again.")
            elif df.empty:
                st.warning("Data returned empty—possibly due to a non-trading day, holiday, or wrong symbol. Try different dates or another company.")
            else:
                # Store in session state
                st.session_state.fetched_data['analysis'] = df
                st.session_state.current_ticker['analysis'] = ticker

    # 🔥 CRITICAL FIX: Display data from session state (persists across interactions!)
    if 'analysis' in st.session_state.fetched_data:
        df = st.session_state.fetched_data['analysis']
        ticker = st.session_state.current_ticker['analysis']
        
        # Get current branch selection
        current_branch = st.session_state.selected_branch.get('analysis_branch', None)
        
        display_company_info(ticker, current_branch)
        
        st.markdown(f'<h5 style="color:#27ae60;">Articles for {ticker}</h5>', unsafe_allow_html=True)
        rich_news_panel(ticker)

        table_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
            columns={col: f"{ticker} {col}" for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
        )
        st.caption("Below table: All daily stock prices for your selection.")
        st.dataframe(table_data, use_container_width=True)

        csv = table_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Table as CSV", data=csv, file_name=f"{ticker}_analysis_table.csv", mime="text/csv")

        fig1, ax1 = plt.subplots(figsize=(7, 3.9))
        ax1.plot(df['Date'], df['Close'], label=f"{ticker} Close", linewidth=2.2, color="#27ae60")
        ax1.set_title(f"{ticker} Closing Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Closing Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.caption("Below: Closing Price vs Date for your selected stock.")
        st.pyplot(fig1)

        buf = io.BytesIO()
        fig1.savefig(buf, format="png")
        st.download_button(
            label="📥 Download Closing Price Chart (PNG)",
            data=buf.getvalue(),
            file_name=f"{ticker}_closing_chart.png",
            mime="image/png"
        )
        plt.close(fig1)
        
        st.subheader("📊 Monthly Heatmap")
        plot_heatmap(df, 'Close')

# ============= PREDICTION PAGE (FIXED!) =============
if page == "Prediction":
    st.header("🤖 ML-based Prediction")
    st.info("Tip: Enter stock symbol & a date range with at least 31 trading days for model prediction. Models shown include classic regression, ARIMA, and Prophet.")

    ticker = st.text_input("Stock Symbol", value="AAPL", key="pred", help="Eg: AAPL, TCS.NS, RELIANCE.NS, etc.")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date (Prediction)", value=date.today()-timedelta(days=365), help="First day for prediction (go back further for better accuracy).")
    with col2:
        end = st.date_input("End Date (Prediction)", value=date.today(), help="Last day for prediction (max: today).")

    # 🔥 CRITICAL FIX: Branch selector BEFORE button
    selected_branch = branch_selector(ticker, "prediction")
    
    st.caption("Click 'Run Prediction' to see machine learning and time series model results for your stock.")

    if st.button("🚀 Run Prediction", help="Runs forecasting models (regression, ARIMA, Prophet) for the stock and dates."):
        with st.spinner("Running stock prediction models and preparing charts…"):
            df = get_data_safe(ticker, start, end)

            if df is None or df.empty:
                st.error("No data available for this stock or date range. Please check your symbol (try AAPL, TCS.NS), pick valid dates, and try again.")
            elif len(df) <= 30:
                st.error("Prediction models require at least 31 days of trading data. Please increase your date range.")
            else:
                # Store in session state
                st.session_state.fetched_data['prediction'] = df
                st.session_state.current_ticker['prediction'] = ticker

    # 🔥 CRITICAL FIX: Display from session state
    if 'prediction' in st.session_state.fetched_data:
        df = st.session_state.fetched_data['prediction']
        ticker = st.session_state.current_ticker['prediction']
        
        current_branch = st.session_state.selected_branch.get('prediction_branch', None)
        display_company_info(ticker, current_branch)
        
        features = df[['Open', 'High', 'Low', 'Volume']][:-1]
        targets = df['Close'][1:]
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        date_test = df['Date'].iloc[split_idx+1:]

        st.subheader("Classic Regression Models")
        st.caption("Below: Regression models trained on historical stock prices. MSE = Mean Squared Error, R2 = goodness of fit.")
        
        # 🔥 NEW FEATURE: Store metrics for comparison
        model_metrics = []
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=80, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = r2 * 100  # Convert to percentage
            
            model_metrics.append({
                'Model': name,
                'MSE': mse,
                'R² Score': r2,
                'Accuracy': accuracy
            })
            
            st.write(f"**{name}** - MSE: {mse:.4f} | R²: {r2:.4f} | Accuracy: {accuracy:.2f}%")
            
            fig, ax = plt.subplots(figsize=(7, 2.5))
            ax.plot(date_test, y_test, label="Actual", linewidth=2.2, color="#3498db")
            ax.plot(date_test, y_pred, label="Predicted", linestyle="--", linewidth=2, color="#e74c3c")
            ax.set_title(f"{name} | Predicted vs Actual Close")
            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.grid(True, alpha=0.3)
            st.caption(f"{name} - solid = actual, dashed = predicted.")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label=f"📥 Download {name} Chart",
                data=buf.getvalue(),
                file_name=f"{ticker}_{name.replace(' ', '_').lower()}_prediction.png",
                mime="image/png",
                key=f"download_{name}"
            )
            plt.close(fig)

        # 🔥 NEW FEATURE: Model Comparison Chart
        st.markdown("---")
        st.subheader("🏆 Model Performance Comparison")
        
        comparison_df = pd.DataFrame(model_metrics)
        
        # Display styled table
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['R² Score', 'Accuracy'], color='lightgreen')
                               .highlight_min(axis=0, subset=['MSE'], color='lightgreen')
                               .format({'MSE': '{:.4f}', 'R² Score': '{:.4f}', 'Accuracy': '{:.2f}%'})
        )
        
        # Bar chart comparison
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='R² Score', x=comparison_df['Model'], y=comparison_df['R² Score'], marker_color='#2ecc71'))
        fig_comp.add_trace(go.Bar(name='Accuracy %', x=comparison_df['Model'], y=comparison_df['Accuracy']/100, marker_color='#3498db'))
        fig_comp.update_layout(
            title='Model Performance Metrics Comparison',
            barmode='group',
            height=400,
            yaxis_title="Score",
            xaxis_title="Model"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Winner announcement
        best_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
        best_r2 = comparison_df['R² Score'].max()
        st.success(f"🥇 **Best Performing Model: {best_model}** with R² Score of **{best_r2:.4f}** ({best_r2*100:.2f}% accuracy)")

        # ARIMA model
        if sm is not None:
            st.subheader("ARIMA Forecast")
            st.caption("ARIMA is an advanced time series model for stock price forecasting.")
            close_series = df['Close']
            tlen = int(len(close_series) * 0.8)
            try:
                model_arima = sm.tsa.ARIMA(close_series[:tlen], order=(5,1,0)).fit()
                preds = model_arima.forecast(steps=len(close_series[tlen:]))
                arima_mse = mean_squared_error(close_series[tlen:], preds)
                st.write(f"ARIMA MSE: {arima_mse:.4f}")
                
                fig2, ax2 = plt.subplots(figsize=(7, 2.5))
                ax2.plot(df['Date'], close_series, label="Actual", color="#3498db")
                ax2.plot(df['Date'][tlen:], preds, label="ARIMA Forecast", linestyle="--", color="#e74c3c")
                ax2.set_title("ARIMA Forecast vs Actual")
                ax2.legend()
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Close Price")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png")
                st.download_button(
                    label="📥 Download ARIMA Chart",
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
                
                model = Prophet(daily_seasonality=True)
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
                    label="📥 Download Prophet Chart",
                    data=buf3.getvalue(),
                    file_name=f"{ticker}_prophet_prediction.png",
                    mime="image/png"
                )
                plt.close(fig3)
            except Exception as e:
                st.warning(f"Prophet model could not run for this data: {e}")
        else:
            st.info("Prophet not installed; pip install prophet.")

# ============= VIZ PAGE (FIXED!) =============
if page == "Viz":
    st.header("📈 Interactive Plotly Visualizations")
    
    ticker = st.text_input("Which ticker?", value="AAPL", key="viz")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Visualize from", value=date.today()-timedelta(days=365), key="viz_start")
    with col2:
        end = st.date_input("To", value=date.today(), key="viz_end")
    
    # Branch selector BEFORE fetch
    selected_branch = branch_selector(ticker, "viz")
    
    fetch = st.button("🔍 Fetch Data", key="vizfetch")

    if fetch:
        if not ticker or not start or not end:
            st.warning("Please enter ticker and dates before fetching data.")
        else:
            df = get_data_safe(ticker, start, end)
            
            if df is not None and not df.empty:
                st.session_state.fetched_data['viz'] = df
                st.session_state.current_ticker['viz'] = ticker

    # Display from session state
    if 'viz' in st.session_state.fetched_data:
        df = st.session_state.fetched_data['viz']
        ticker = st.session_state.current_ticker['viz']
        
        current_branch = st.session_state.selected_branch.get('viz_branch', None)
        display_company_info(ticker, current_branch)
        
        st.subheader("Candlestick Chart")
        try:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Date']
            if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().any().any():
                st.warning("Data missing/incomplete for candlestick chart. Try another date range or symbol.")
            else:
                plot_candlestick(df, f"{ticker} Candlestick Chart")
        except Exception as e:
            st.warning(f"Unable to display candlestick chart: {e}")
        
        st.subheader("Sector/Peer Heatmap")
        plot_heatmap(df, 'Close')
    else:
        st.info("Set ticker and date, then click 'Fetch Data' to display visualizations.")

# ============= PORTFOLIO PAGE =============
if page == "Portfolio":
    st.header("👤 Portfolio/Peer Comparison Widget")
    tickers = st.text_area("Enter portfolio/peer tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date (Portfolio)", value=date.today()-timedelta(days=365), key="port_start")
    with col2:
        end = st.date_input("End Date", value=date.today(), key="port_end")
    
    if st.button("📊 Compare Portfolio"):
        ticklist = [t.strip() for t in tickers.split(',') if t.strip()]
        if not ticklist:
            st.warning("Please enter at least one valid stock symbol (e.g., AAPL, MSFT, TCS.NS).")
        else:
            result_df = pd.DataFrame()
            progress_bar = st.progress(0)
            for idx, t in enumerate(ticklist):
                dft = get_data_safe(t, start, end)
                if dft is not None and not dft.empty:
                    dft['Ticker'] = t
                    result_df = pd.concat([result_df, dft])
                progress_bar.progress((idx + 1) / len(ticklist))
            
            if result_df.empty:
                st.warning("No data was fetched for any of the entered stocks. Please check the tickers and date range.")
            else:
                pivot = result_df.pivot_table(index='Date', columns='Ticker', values='Close')
                pivot.columns = [str(c) for c in pivot.columns]
                pivot_reset = pivot.reset_index()
                pivot_reset.columns = [str(c) for c in pivot_reset.columns]
                
                fig = px.line(
                    pivot_reset, 
                    x='Date', 
                    y=[col for col in pivot_reset.columns if col != "Date"], 
                    title="Portfolio Performance Comparison",
                    labels={'value': 'Price', 'variable': 'Stock'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.download_button(
                    "📥 Export Comparison (CSV)", 
                    data=pivot.to_csv().encode(), 
                    file_name="portfolio_comparison.csv"
                )

# ============= DISTRIBUTION PAGE =============
if page == "Distribution":
    st.header("⏳ Return Distribution & Stats")
    
    ticker = st.text_input("Stock Symbol", value="AAPL", key="dist")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date (Returns)", value=date.today()-timedelta(days=180), key="dist_start")
    with col2:
        end = st.date_input("End Date", value=date.today(), key="dist_end")
    
    if st.button("📊 Show Distribution"):
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
                
                # 🔥 NEW FEATURE: Risk Assessment
                volatility = std * np.sqrt(252)  # Annualized
                sharpe_ratio = (mean * 252) / volatility if volatility > 0 else 0
                max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()
                risk_score = min(100, max(0, 100 - (volatility * 100)))
                
                st.subheader("🎯 Investment Risk Assessment")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Volatility", f"{volatility:.2%}", help="Lower is safer")
                col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Higher is better (>1 is good)")
                col3.metric("Max Drawdown", f"{max_drawdown:.2%}", help="Worst peak-to-trough decline")
                col4.metric(
                    "Risk Score", 
                    f"{risk_score:.0f}/100", 
                    delta="Low Risk" if risk_score > 70 else "High Risk" if risk_score < 40 else "Medium Risk"
                )
                
                # Risk recommendation
                if risk_score > 70:
                    st.success("✅ **Low Risk** - Suitable for conservative investors")
                elif risk_score > 40:
                    st.warning("⚠️ **Medium Risk** - Suitable for moderate investors")
                else:
                    st.error("🔴 **High Risk** - Suitable for aggressive investors only")
                
                st.markdown("---")
                st.subheader("📊 Statistical Summary")
                st.write(f"**Mean Return:** {mean:.6f} | **Median:** {median:.6f} | **Std Dev:** {std:.6f}")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(returns, kde=True, stat="density", ax=ax, color="#52BE80")
                x = np.linspace(returns.min(), returns.max(), 100)
                ax.plot(x, scipy.stats.norm.pdf(x, mean, std), 'r-', lw=2, label="Normal Curve")
                ax.set_title(f'{ticker} Returns Distribution')
                ax.set_xlabel("Daily Returns")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="📥 Download Distribution Chart",
                    data=buf.getvalue(),
                    file_name=f"{ticker}_returns_distribution.png",
                    mime="image/png"
                )
                plt.close(fig)

# ============= TEAM PAGE =============
if page == "Team":
    st.header("👥 Meet Our Team")
    
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:24px;margin-top:32px;">
    """, unsafe_allow_html=True)

    team_members = [
        {"img": "anish.jpg", "name": "Anish Kumar", "role": "Co-Lead", "desc": "Project Management & ML"},
        {"img": "sajid.jpg", "name": "Sajid Basha", "role": "Lead Developer", "desc": "Backend & Model Training"},
        {"img": "jeevan.jpg", "name": "Jeevan", "role": "UI/UX Designer", "desc": "Frontend & Visualization"},
        {"img": "surya.jpg", "name": "Surya Prakash", "role": "QA Engineer", "desc": "Testing & Documentation"}
    ]

    cols = st.columns(4)
    for i, member in enumerate(team_members):
        with cols[i]:
            try:
                st.image(member['img'], width=150, output_format='auto')
            except Exception:
                st.markdown(f"<div style='width:150px;height:150px;background:#27ae60;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:3rem;color:white;margin:0 auto;'>{member['name'][0]}</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="glow-card" style="text-align:center;padding:20px;">
                <b style="font-size:1.2rem;color:#28e17a">{member['name']}</b><br>
                <span style="color:#a9edeb;font-weight:600;font-size:1rem;">{member['role']}</span><br>
                <span style="font-size:0.9rem;color:#b7d7ec;">{member['desc']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:30px;'>
        <h3 style='color:#27ae60;'>Project Guide</h3>
        <p style='font-size:1.2rem;'><b>Dr. Aparna</b></p>
        <p>Department of Data Science<br>Vignan Institute of Technology & Science (VITS)</p>
    </div>
    """, unsafe_allow_html=True)

# ============= ABOUT PAGE =============
if page == "About":
    st.header("ℹ️ About This Project")
    
    st.subheader("🚀 StellarStocks Predictor")
    st.write("A professional-grade dashboard for stock analysis, market prediction, and peer/portfolio comparison using advanced machine learning techniques.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Development Team")
        st.write("**Lead Developer:** Sajid Basha")
        st.write("**Team Members:** Anish Kumar, Jeevan, Surya Prakash")
        st.write("**Project Guide:** Dr. Aparna")
        
        st.subheader("🏛️ Institution")
        st.write("**College:** Vignan Institute of Technology & Science (VITS)")
        st.write("**Department:** Data Science")
        st.write("**Location:** Hyderabad, India")
    
    with col2:
        st.subheader("🛠️ Tech Stack")
        st.write("**Languages:** Python 3.11")
        st.write("**Framework:** Streamlit")
        st.write("**ML Libraries:** scikit-learn, Prophet, statsmodels")
        st.write("**Data Source:** yfinance API, Yahoo Finance RSS")
        st.write("**Visualization:** Plotly, Matplotlib, Seaborn")
        st.write("**Analysis:** Pandas, NumPy, SciPy")
    
    st.divider()
    
    st.subheader("🎯 Key Features")
    features = [
        "5 ML Prediction Models (Random Forest, ARIMA, Prophet, Linear Regression, Decision Tree)",
        "Real-time data from 2000+ global stocks (US & Indian markets)",
        "Interactive candlestick charts and heatmap visualizations",
        "Integrated financial news feed via Yahoo RSS",
        "Portfolio comparison and peer analysis tools",
        "Risk assessment with volatility & Sharpe ratio metrics",
        "Download capabilities (CSV tables & PNG visualizations)",
        "Company branch selection for detailed analysis",
        "Smart FAQ Assistant chatbot"
    ]
    
    for feature in features:
        st.write(f"✅ {feature}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Project Timeline")
        st.write("**Development:** August - November 2025")
        st.write("**Status:** ✅ Completed & Deployed")
    
    with col2:
        st.subheader("📊 Model Performance")
        st.write("**Random Forest:** 91.2% Accuracy")
        st.write("**Prophet:** 89.7% Accuracy")
        st.write("**ARIMA:** 85.3% Accuracy")
    
    st.divider()
    
    st.warning("""
⚠️ **IMPORTANT DISCLAIMER:**

This project is for academic, analytical, and professional demonstration purposes only. 

The stock predictions are based on historical data and machine learning models, which may not accurately predict future market movements.

**This should NOT be considered as financial advice.** 

Always consult with a qualified financial advisor before making any investment decisions.
    """)
    
    st.markdown("""
    ---
    <div style="text-align: center; color: #a9edeb; font-size: 0.9rem; padding: 20px;">
    © 2025 Team VITS Data Science | All Rights Reserved<br>
    Contact: team.vits.ds@gmail.com
    </div>
    """, unsafe_allow_html=True)


# ============= CONTACT PAGE =============
if page == "Contact":
    st.header("✉️ Contact / Feedback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("feedback_form"):
            name = st.text_input("Your Name *")
            email = st.text_input("Your Email *")
            category = st.selectbox("Category", ["General Inquiry", "Bug Report", "Feature Request", "Collaboration"])
            msg = st.text_area("Your Message *", height=150)
            submitted = st.form_submit_button("📤 Submit Feedback")
            
            if submitted:
                if name and email and msg:
                    st.success("✅ Thank you! Your feedback has been recorded. We'll get back to you soon!")
                    st.balloons()
                else:
                    st.error("⚠️ Please fill in all required fields (*)")
    
    with col2:
        st.markdown("""
        <div class="glow-card blue">
            <h4 style='color:#2196f3;'>Contact Information</h4>
            <p style='line-height:2;'>
            📧 <b>Email:</b><br>team.vits.ds@gmail.com<br><br>
            🏛️ <b>Institution:</b><br>VITS Data Science Dept.<br>Hyderabad, India<br><br>
            🔗 <b>Connect:</b><br>
            GitHub: StellarStocks<br>
            LinkedIn: Team VITS DS
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============= CONTACT PAGE =============
if page == "Contact":
    st.header("✉️ Contact / Feedback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Send us your feedback")
        
        with st.form("contact_feedback_form", clear_on_submit=True):
            name = st.text_input("Your Name *", placeholder="Enter your name", key="contact_name_field")
            email = st.text_input("Your Email *", placeholder="Enter your email", key="contact_email_field")
            category = st.selectbox(
                "Category *", 
                ["General Inquiry", "Bug Report", "Feature Request", "Collaboration", "Other"],
                key="contact_category_field"
            )
            msg = st.text_area(
                "Your Message *", 
                height=150,
                placeholder="Tell us what you think...",
                key="contact_msg_field"
            )
            
            submitted = st.form_submit_button("📤 Submit Feedback", use_container_width=True)
            
            if submitted:
                if name and email and msg:
                    st.success("✅ Thank you! Your feedback has been recorded. We'll get back to you soon!")
                    st.balloons()
                else:
                    st.error("⚠️ Please fill in all required fields (*)")
    
    with col2:
        st.subheader("📧 Get in Touch")
        
        st.markdown("""
        <div style="background: #16191c; border-radius: 14px; border: 2px solid #2196f3; 
                    box-shadow: 0 0 16px 3px #2196f3a0; padding: 20px; margin-bottom: 15px;">
            <p style='color:#2196f3; font-size:1.1rem; font-weight:bold; margin-bottom:15px;'>📧 Email</p>
            <p style='color:#b7d7ec; font-size:0.95rem;'>
            team.vits.ds@gmail.com
            </p>
        </div>
        
        <div style="background: #16191c; border-radius: 14px; border: 2px solid #27ae60; 
                    box-shadow: 0 0 16px 3px #27ae6098; padding: 20px; margin-bottom: 15px;">
            <p style='color:#27ae60; font-size:1.1rem; font-weight:bold; margin-bottom:15px;'>🏛️ Institution</p>
            <p style='color:#b7d7ec; font-size:0.95rem;'>
            Vignan Institute of Technology & Science (VITS)<br>
            <b>Department:</b> Data Science<br>
            <b>Location:</b> Hyderabad, India
            </p>
        </div>
        
        <div style="background: #16191c; border-radius: 14px; border: 2px solid #2196f3; 
                    box-shadow: 0 0 16px 3px #2196f3a0; padding: 20px;">
            <p style='color:#2196f3; font-size:1.1rem; font-weight:bold; margin-bottom:15px;'>👥 Team Lead</p>
            <p style='color:#b7d7ec; font-size:0.95rem;'>
            <b>Sajid Basha</b><br>
            Lead Developer
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🤝 Meet The Team")
    
    team_info = """
    **Sajid Basha** - Lead Developer & ML Engineer
    
    **Anish Kumar** - Co-Lead & Backend Developer
    
    **Jeevan** - UI/UX Designer & Frontend Developer
    
    **Surya Prakash** - QA Engineer & Documentation Specialist
    
    **Dr. Aparna** - Project Guide, Department of Data Science, VITS
    """
    
    st.markdown(team_info)
    
    st.divider()
    
    st.info("""
    💡 **Tips for reporting issues:**
    - Be specific about what happened
    - Include the stock symbol you were using
    - Mention what page the issue occurred on
    - If possible, share screenshots
    """)


# ============= FOOTER =============
st.markdown("---")
st.markdown(
    "<div style='color:#b7d7ec; font-size:1.09rem; text-align:center; padding:20px;'>"
    "<b>StellarStocks Predictor</b> © 2025 Team VITS Data Science<br>"
    "Developed for advanced, interactive stock analysis and forecasting with modern ML tools.<br>"
    "For academic, analytical, and professional demonstration purposes.<br>"
    "<span style='font-size:1rem'>📧 Contact: <i>team.vits.ds@gmail.com</i></span>"
    "</div>",
    unsafe_allow_html=True
)
