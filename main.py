# main.py

import yfinance as yf
import pandas as pd

# --- Step 1: Ask user for a stock symbol ---
stock_symbol = input("Enter Stock Symbol (e.g., TCS.NS, RELIANCE.NS, AAPL): ").upper()

# --- Step 2: Fetch historical data from Yahoo Finance ---
df = yf.download(stock_symbol, period='1y', interval='1d')

if df.empty:
    print("No data found. Please check the stock symbol and your internet connection.")
else:
    print("\n--- Data Fetch Successful! ---")
    print(df.head())  # Show first 5 rows as a preview

    # Optionally, save a backup for analysis later:
    df.to_csv(f"data/{stock_symbol}_historical.csv")
    print(f"\nData saved to data/{stock_symbol}_historical.csv")

    # --- Step 3: Statistical Analysis ---
    print("\n--- Descriptive Statistics ---")
    print(df.describe().round(2))

    # Use first column of MultiIndex dataframe (yfinance output)
    close = df['Close'].iloc[:,0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    open_price = df['Open'].iloc[:,0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

    print(f"\nMean Closing Price: {close.mean():.2f}")
    print(f"Median Closing Price: {close.median():.2f}")
    print(f"Variance of Closing Price: {close.var():.2f}")
    print(f"Standard Deviation: {close.std():.2f}")
    print(f"Covariance between Open and Close: {open_price.cov(close):.2f}")
    print(f"Correlation between Open and Close: {open_price.corr(close):.2f}")

    print(f"\nDate Range: {df.index.min().date()} to {df.index.max().date()}")


import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure images folder exists
os.makedirs('images', exist_ok=True)

# --- 4A: Closing Price Trend ---
plt.figure(figsize=(10,5))
plt.plot(close, label='Close Price')
plt.title(f"{stock_symbol} Closing Price Trend (Last 1 Year)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f"images/{stock_symbol}_close_trend.png")
plt.show()

# --- 4B: Moving Averages ---
ma_10 = close.rolling(window=10).mean()
ma_50 = close.rolling(window=50).mean()

plt.figure(figsize=(10,5))
plt.plot(close, label='Close Price')
plt.plot(ma_10, label='10-Day MA', linestyle='--')
plt.plot(ma_50, label='50-Day MA', linestyle=':')
plt.title(f"{stock_symbol} Moving Averages (10 & 50 Days)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f"images/{stock_symbol}_moving_averages.png")
plt.show()

# --- 4C: Correlation Heatmap ---
plt.figure(figsize=(6,4))
corr = df.corr() if isinstance(df, pd.DataFrame) else None
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title(f"{stock_symbol} Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"images/{stock_symbol}_corr_heatmap.png")
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- Prepare the dataset for ML model (predict next day's close) ---
# Use "Close" price, shift by 1 to get 'next day' as the target
X = close.values[:-1].reshape(-1, 1)  # All days except last
y = close.values[1:]                  # Next day's price (target), shifts by 1

# Train/test split (80:20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Train Linear Regression Model ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# --- Evaluate model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Model Results ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# --- Plot: Actual vs Predicted ---
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y_test)), y_test, label='Actual', color='b')
plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', color='r', linestyle='--')
plt.title(f"{stock_symbol} Actual vs Predicted Closing Price")
plt.xlabel('Test Sample')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f"images/{stock_symbol}_actual_vs_predicted.png")
plt.show()

# --- Predict next day's close (extrapolate) ---
last_close = close.values[-1].reshape(1, -1)
next_day_pred = lr.predict(last_close)
print(f"\nPredicted closing price for next day: {next_day_pred[0]:.2f}")
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- Prepare Features for Multiple Regression ---
features = df[['Open', 'High', 'Low', 'Volume']]
if isinstance(features, pd.DataFrame) and isinstance(close, pd.Series):
    features = features.iloc[:-1]
    targets = close[1:]
else:
    # fallback just in case: single-level columns
    features = df[['Open', 'High', 'Low', 'Volume']][:-1]
    targets = df['Close'][1:]

# --- Train/Test Split ---
split_idx = int(len(features) * 0.8)
X_train, X_test = features[:split_idx], features[split_idx:]
y_train, y_test = targets[:split_idx], targets[split_idx:]

results = {}

# --- Linear Regression (Multiple Features) ---
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_pred_lr = lr_multi.predict(X_test)
results['Linear Regression'] = (y_test, y_pred_lr)

# --- Decision Tree Regression ---
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = (y_test, y_pred_dt)

# --- Random Forest Regression ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = (y_test, y_pred_rf)

# --- Print and Plot Results ---
for name, (y_true, y_pred) in results.items():
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(y_true)), y_true, label='Actual')
    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', linestyle='--')
    plt.title(f"{stock_symbol} Actual vs Predicted ({name})")
    plt.xlabel('Test Sample')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    img_path = f"images/{stock_symbol}_{name.replace(' ', '_').lower()}_actual_vs_pred.png"
    plt.savefig(img_path)
    plt.show()



from fpdf import FPDF
import glob

def create_pdf_report(stock_symbol):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Stock Market Prediction Report: {stock_symbol}", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    
    # Description
    pdf.multi_cell(0, 10, 
        "This report summarizes the statistical analysis, visual trends, and machine learning model predictions for the selected stock's prices."
    )
    
    # Insert statistics
    pdf.ln(5)
    pdf.cell(0, 10, "Summary Statistics:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Mean Closing Price: {close.mean():.2f}", ln=1)
    pdf.cell(0, 8, f"Median Closing Price: {close.median():.2f}", ln=1)
    pdf.cell(0, 8, f"Variance: {close.var():.2f}", ln=1)
    pdf.cell(0, 8, f"Standard Deviation: {close.std():.2f}", ln=1)
    pdf.cell(0, 8, f"Date Range: {df.index.min().date()} to {df.index.max().date()}", ln=1)
    pdf.ln(5)

    # Add images
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Visualizations and Model Results:", ln=1)
    img_files = sorted(glob.glob(f"images/{stock_symbol}_*.png"))
    for img in img_files:
        pdf.image(img, w=180)
        pdf.ln(8)

    # Save PDF
    report_path = f"data/{stock_symbol}_report.pdf"
    pdf.output(report_path)
    print(f"PDF report generated: {report_path}")

# Call PDF creation at end (after all analysis & images saved)
create_pdf_report(stock_symbol)
