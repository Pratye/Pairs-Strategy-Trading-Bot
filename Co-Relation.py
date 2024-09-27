import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def stock_correlation_and_hedge_ratio(stock1, stock2, start_date, end_date):
    """
    Downloads stock data, calculates correlation and hedge ratio between two stocks,
    and plots the adjusted closing prices and the spread.

    Parameters:
    - stock1: Ticker symbol for the first stock (e.g., 'AAPL')
    - stock2: Ticker symbol for the second stock (e.g., 'MSFT')
    - start_date: Start date for historical data (YYYY-MM-DD)
    - end_date: End date for historical data (YYYY-MM-DD)

    Returns:
    - correlation: Pearson correlation coefficient between the two stocks
    - hedge_ratio: Calculated hedge ratio between the two stocks
    """

    # Download stock data from Yahoo Finance
    data1 = yf.download(stock1, start=start_date, end=end_date)['Adj Close']
    data2 = yf.download(stock2, start=start_date, end=end_date)['Adj Close']

    # Create a DataFrame with both stocks
    df = pd.DataFrame({stock1: data1, stock2: data2})

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Calculate Pearson correlation
    correlation = df[stock1].corr(df[stock2])
    print(f"Correlation between {stock1} and {stock2}: {correlation:.4f}")

    # Calculate Hedge Ratio using linear regression (stock1 = beta * stock2 + alpha)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[stock2], df[stock1])
    hedge_ratio = slope
    print(f"Hedge Ratio (beta) between {stock1} and {stock2}: {hedge_ratio:.4f}")

    # Calculate the spread
    df['Spread'] = df[stock1] - hedge_ratio * df[stock2]

    # Plot Adjusted Close Prices
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df[stock1], label=stock1)
    plt.plot(df.index, df[stock2], label=stock2)
    plt.title(f"{stock1} vs {stock2} Adjusted Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    # Plot Spread
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Spread'], label='Spread', color='purple')
    plt.axhline(df['Spread'].mean(), color='red', linestyle='--', label='Mean Spread')
    plt.title("Spread Between Stocks")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return correlation, hedge_ratio


# Example usage
if __name__ == "__main__":
    stock1 = 'TSM'  # Example stock 1
    stock2 = 'NVDA'  # Example stock 2
    start_date = '2024-09-17'
    end_date = '2024-09-27'

    correlation, hedge_ratio = stock_correlation_and_hedge_ratio(stock1, stock2, start_date, end_date)
