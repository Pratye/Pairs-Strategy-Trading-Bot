import os
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pytz import timezone

# ----------------------------
# Configuration and Setup
# ----------------------------

# Define the stock symbols
symbol1 = "TSM"  # Taiwan Semiconductor Manufacturing Company
symbol2 = "NVDA"  # NVIDIA Corporation

# Define the historical period for backtesting
# For 5-minute intervals over 6 months
backtest_period = "1mo"
backtest_interval = "30m"

# Define Z-score thresholds
entry_zscore = 1  # Threshold to enter a trade
exit_zscore = 0.5  # Threshold to exit a trade

# Capital allocation parameters
initial_capital = 100000  # Starting capital in USD
capital_allocation = 0.1  # 10% of available capital per trade

# Define timezone
tz = timezone('US/Eastern')


# ----------------------------
# Fetch Historical Data
# ----------------------------

def fetch_historical_data(symbol, period, interval):
    """
    Fetches historical price data for a given symbol using yfinance.
    Args:
        symbol (str): Stock symbol.
        period (str): Period for which to fetch data (e.g., '6mo', '1y').
        interval (str): Data interval (e.g., '1m', '5m').
    Returns:
        pd.DataFrame: DataFrame containing historical price data.
    """
    try:
        stock = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=True)
        if stock.empty:
            print(f"No data fetched for {symbol}. Check the symbol and interval.")
            return pd.DataFrame()
        return stock
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


# Fetch data for both symbols
data1 = fetch_historical_data(symbol1, backtest_period, backtest_interval)
data2 = fetch_historical_data(symbol2, backtest_period, backtest_interval)

# Check if data was fetched successfully
if data1.empty or data2.empty:
    print("Failed to fetch data for one or both symbols. Exiting program.")
    exit()


# ----------------------------
# Data Preparation
# ----------------------------

def align_data(df1, df2):
    """
    Aligns two DataFrames on their datetime indices.
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
    Returns:
        pd.DataFrame: Combined DataFrame with aligned data.
    """
    combined = pd.concat([df1['Close'], df2['Close']], axis=1, join='inner')
    combined.columns = [f'{symbol1}_Close', f'{symbol2}_Close']
    return combined


combined_data = align_data(data1, data2)


# ----------------------------
# Calculate Hedge Ratio (Beta)
# ----------------------------

def calculate_hedge_ratio(y, x):
    """
    Calculates the hedge ratio (beta) using Ordinary Least Squares (OLS) regression.
    Args:
        y (pd.Series): Dependent variable (symbol1 prices).
        x (pd.Series): Independent variable (symbol2 prices).
    Returns:
        float: Hedge ratio (beta).
    """
    x = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    return beta


beta = calculate_hedge_ratio(combined_data[f'{symbol1}_Close'], combined_data[f'{symbol2}_Close'])
print(f"Hedge Ratio (Beta) between {symbol1} and {symbol2}: {beta:.4f}")


# ----------------------------
# Calculate Spread and Z-score
# ----------------------------

def calculate_spread_zscore(df, beta, window=30):
    """
    Calculates the spread and its Z-score.
    Args:
        df (pd.DataFrame): Combined DataFrame with both symbols' close prices.
        beta (float): Hedge ratio.
        window (int): Rolling window size for mean and std deviation.
    Returns:
        pd.DataFrame: DataFrame with spread and Z-score.
    """
    df['Spread'] = df[f'{symbol1}_Close'] - beta * df[f'{symbol2}_Close']
    df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
    df['Spread_Std'] = df['Spread'].rolling(window=window).std()
    df['Zscore'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
    return df


combined_data = calculate_spread_zscore(combined_data, beta)

# Drop rows with NaN values resulting from rolling calculations
combined_data.dropna(inplace=True)


# ----------------------------
# Backtesting Simulation
# ----------------------------

def backtest_pairs_trading(df, beta, initial_capital, capital_allocation, entry_zscore, exit_zscore):
    """
    Simulates the pairs trading strategy on historical data.
    Args:
        df (pd.DataFrame): DataFrame with spread and Z-score.
        beta (float): Hedge ratio.
        initial_capital (float): Starting capital in USD.
        capital_allocation (float): Fraction of capital to allocate per trade.
        entry_zscore (float): Z-score threshold to enter a trade.
        exit_zscore (float): Z-score threshold to exit a trade.
    Returns:
        pd.DataFrame: Portfolio value over time.
    """
    capital = initial_capital
    portfolio = []
    position = None  # None, 'long', or 'short'
    qty1 = 0
    qty2 = 0

    for idx, row in df.iterrows():
        zscore = row['Zscore']
        price1 = row[f'{symbol1}_Close']
        price2 = row[f'{symbol2}_Close']

        # Entry Conditions
        if position is None:
            if zscore > entry_zscore:
                # Short symbol1 and Long symbol2
                allocated_funds = capital * capital_allocation
                allocated_funds_symbol1 = allocated_funds / 2
                allocated_funds_symbol2 = allocated_funds / 2

                qty1 = int(allocated_funds_symbol1 / price1)
                qty2 = int((allocated_funds_symbol2 * beta) / price2)

                if qty1 > 0 and qty2 > 0:
                    position = 'short'
                    capital -= (qty1 * price1)  # Shorting symbol1
                    capital += (qty2 * price2)  # Longing symbol2
                    print(f"{idx}: Enter SHORT {symbol1} ({qty1} shares) and LONG {symbol2} ({qty2} shares)")
            elif zscore < -entry_zscore:
                # Long symbol1 and Short symbol2
                allocated_funds = capital * capital_allocation
                allocated_funds_symbol1 = allocated_funds / 2
                allocated_funds_symbol2 = allocated_funds / 2

                qty1 = int(allocated_funds_symbol1 / price1)
                qty2 = int((allocated_funds_symbol2 * beta) / price2)

                if qty1 > 0 and qty2 > 0:
                    position = 'long'
                    capital -= (qty2 * price2)  # Shorting symbol2
                    capital += (qty1 * price1)  # Longing symbol1
                    print(f"{idx}: Enter LONG {symbol1} ({qty1} shares) and SHORT {symbol2} ({qty2} shares)")
        else:
            # Exit Conditions
            if abs(zscore) < exit_zscore:
                if position == 'short':
                    # Close SHORT symbol1 and LONG symbol2
                    capital += (qty1 * price1)  # Covering short symbol1
                    capital += (qty2 * price2)  # Selling long symbol2
                    print(f"{idx}: Exit SHORT {symbol1} and LONG {symbol2} ({qty1} shares, {qty2} shares)")
                elif position == 'long':
                    # Close LONG symbol1 and SHORT symbol2
                    capital -= (qty1 * price1)  # Selling long symbol1
                    capital -= (qty2 * price2)  # Covering short symbol2
                    print(f"{idx}: Exit LONG {symbol1} and SHORT {symbol2} ({qty1} shares, {qty2} shares)")
                position = None
                qty1 = 0
                qty2 = 0

        # Append current capital to portfolio
        portfolio.append(capital)

    # Create a DataFrame for portfolio value over time
    portfolio_df = pd.DataFrame(data=portfolio, index=df.index, columns=['Portfolio_Value'])
    return portfolio_df


# Run backtest
portfolio = backtest_pairs_trading(
    df=combined_data,
    beta=beta,
    initial_capital=initial_capital,
    capital_allocation=capital_allocation,
    entry_zscore=entry_zscore,
    exit_zscore=exit_zscore
)


# ----------------------------
# Performance Metrics
# ----------------------------

def calculate_performance(portfolio_df):
    """
    Calculates performance metrics for the backtested portfolio.
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio value over time.
    """
    portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
    cumulative_return = (portfolio_df['Portfolio_Value'][-1] / portfolio_df['Portfolio_Value'][0]) - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_df)) - 1  # Assuming 252 trading days
    sharpe_ratio = portfolio_df['Returns'].mean() / portfolio_df['Returns'].std() * np.sqrt(
        252)  # Risk-free rate assumed 0

    max_drawdown = ((portfolio_df['Portfolio_Value'].cummax() - portfolio_df['Portfolio_Value']) / portfolio_df[
        'Portfolio_Value'].cummax()).max()

    print("\n--- Performance Metrics ---")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")


# Calculate performance
calculate_performance(portfolio)


# ----------------------------
# Visualization
# ----------------------------

def plot_performance(portfolio_df):
    """
    Plots the portfolio value over time.
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio value over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df['Portfolio_Value'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Drawdown
    portfolio_df['Cumulative_Max'] = portfolio_df['Portfolio_Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Cumulative_Max'] - portfolio_df['Portfolio_Value']) / portfolio_df[
        'Cumulative_Max']

    plt.figure(figsize=(12, 4))
    plt.plot(portfolio_df['Drawdown'], label='Drawdown', color='red')
    plt.title('Portfolio Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot performance
plot_performance(portfolio)

# ----------------------------
# Summary
# ----------------------------

print("\nBacktesting Completed Successfully!")
