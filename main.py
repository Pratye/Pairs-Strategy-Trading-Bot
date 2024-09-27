import os
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import alpaca_trade_api as tradeapi
import schedule
import time
import logging
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv

# ----------------------------
# Configuration and Setup
# ----------------------------

# Load environment variables from .env file
load_dotenv()

# Alpaca API credentials
API_KEY = 'PKCWAF1UPW4BS6BRIFG8'
SECRET_KEY = 'nQXCChDtIktf5VdbyPzFApLLRp1gMLmHfpEfogZw'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Trading Pair Configuration
symbol1 = "TSM"   # Taiwan Semiconductor Manufacturing Company
symbol2 = "NVDA"  # NVIDIA Corporation

# Initial Capital (for reference; actual capital is fetched from Alpaca account)
initial_capital = 2000  # Starting with $2,000 for demonstration

# Trading Strategy Parameters
entry_zscore = 1    # Z-score threshold to enter a trade
exit_zscore = 0.5     # Z-score threshold to exit a trade
capital_allocation = 0.1  # 10% of available capital per trade

# Historical Data Configuration
data_interval = "30m"        # 5-minute intervals
data_period = "1mo"          # Past 5 trading days
adjusted_close = True       # Use adjusted close prices

# Timezone Configuration
tz = timezone('US/Eastern')  # US Eastern Time

# Position Tracking
open_positions = {
    symbol1: None,
    symbol2: None
}

# ----------------------------
# Logging Configuration
# ----------------------------

logging.basicConfig(
    filename='pairs_trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Console handler for logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ----------------------------
# Helper Functions
# ----------------------------

def fetch_historical_data(symbol, period, interval):
    """
    Fetches historical price data for a given symbol using yfinance.

    Args:
        symbol (str): Stock symbol.
        period (str): Period for which to fetch data (e.g., '5d', '1mo').
        interval (str): Data interval (e.g., '1m', '5m').

    Returns:
        pd.DataFrame: DataFrame containing historical price data.
    """
    try:
        stock = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=adjusted_close)
        if stock.empty:
            logging.error(f"No data fetched for {symbol}. Check the symbol and interval.")
            return pd.DataFrame()
        return stock
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

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
    logging.info(f"Hedge Ratio (Beta) between {symbol1} and {symbol2}: {beta:.4f}")
    return beta

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

def get_available_funds():
    """
    Retrieves the available cash from Alpaca account.

    Returns:
        float: Available cash.
    """
    account = api.get_account()
    return float(account.cash)

def get_position(symbol):
    """
    Checks if there's an open position for a given symbol.

    Args:
        symbol (str): Stock symbol.

    Returns:
        tradeapi.rest.Position or None: Position object if exists, else None.
    """
    try:
        position = api.get_position(symbol)
        return position
    except tradeapi.rest.APIError as e:
        if 'position does not exist' in str(e).lower():
            return None
        else:
            logging.error(f"Error fetching position for {symbol}: {e}")
            return None

def check_open_orders():
    """
    Retrieves any open orders and handles them accordingly.
    """
    try:
        open_orders = api.list_orders(status='open')
        if open_orders:
            logging.info(f"Found {len(open_orders)} open order(s). Reviewing...")
            for order in open_orders:
                logging.info(f"Open Order - ID: {order.id}, Symbol: {order.symbol}, Side: {order.side}, Qty: {order.qty}, Type: {order.type}")
                # Optionally, you can choose to cancel open orders
                # Uncomment the following lines to cancel all open orders
                # api.cancel_order(order.id)
                # logging.info(f"Cancelled Order ID: {order.id}")
        else:
            logging.info("No open orders found.")
    except Exception as e:
        logging.error(f"Error checking open orders: {e}")

def check_open_positions():
    """
    Checks for any existing open positions and updates the open_positions dictionary.
    """
    try:
        positions = api.list_positions()
        if positions:
            logging.info(f"Found {len(positions)} open position(s). Updating position tracking...")
            for pos in positions:
                if pos.symbol == symbol1:
                    open_positions[symbol1] = 'long' if pos.side == 'long' else 'short'
                elif pos.symbol == symbol2:
                    open_positions[symbol2] = 'long' if pos.side == 'long' else 'short'
                logging.info(f"Open Position - Symbol: {pos.symbol}, Side: {pos.side}, Qty: {pos.qty}")
        else:
            logging.info("No open positions found.")
            open_positions[symbol1] = None
            open_positions[symbol2] = None
    except Exception as e:
        logging.error(f"Error checking open positions: {e}")

def initialize_bot():
    """
    Initializes the trading bot by checking for any existing open orders or positions.
    """
    logging.info("Initializing Pairs Trading Bot...")
    check_open_orders()
    check_open_positions()
    logging.info(f"Initial Position States: {open_positions}")

def execute_trade(signal, current_time, hedge_ratio, capital_allocation=0.1):
    """
    Executes trades based on the trading signal while managing capital allocation.

    Args:
        signal (str): Trading signal ('long', 'short', 'exit', or 'hold').
        current_time (pd.Timestamp): Current timestamp.
        hedge_ratio (float): Hedge ratio (beta) between symbol1 and symbol2.
        capital_allocation (float): Fraction of capital to allocate per trade (e.g., 0.1 for 10%).
    """
    global open_positions

    try:
        available_funds = get_available_funds()
        logging.info(f"Available Funds: ${available_funds:.2f}")

        # Fetch current stock prices
        combined_latest = combined_data.loc[current_time]
        symbol1_price = combined_latest[f'{symbol1}_Close']
        symbol2_price = combined_latest[f'{symbol2}_Close']

        # Calculate allocated capital for this trade
        allocated_funds = available_funds * capital_allocation
        logging.info(f"Allocated Funds for Trade: ${allocated_funds:.2f}")

        # Split allocated_funds between two legs
        allocated_funds_symbol1 = allocated_funds / 2
        allocated_funds_symbol2 = allocated_funds / 2

        # Calculate position sizes
        qty1 = max(1, int(allocated_funds_symbol1 / symbol1_price))
        qty2 = max(1, int((allocated_funds_symbol2 * hedge_ratio) / symbol2_price))
        logging.info(f"Position Sizes - {symbol1}: {qty1} shares, {symbol2}: {qty2} shares")

        # Determine required funds
        required_funds = (qty1 * symbol1_price) + (qty2 * symbol2_price * hedge_ratio)
        logging.info(f"Required Funds for Trade: ${required_funds:.2f}")

        # Check if required funds exceed allocated funds
        if required_funds > allocated_funds:
            logging.warning(f"Allocated funds are insufficient for the calculated positions. Adjusting position sizes.")

            # Calculate scaling factor
            scaling_factor = allocated_funds / required_funds

            # Adjust position sizes
            qty1 = max(1, int(qty1 * scaling_factor))
            qty2 = max(1, int(qty2 * scaling_factor))
            logging.info(f"Adjusted Position Sizes - {symbol1}: {qty1} shares, {symbol2}: {qty2} shares")

            # Recalculate required funds after adjustment
            required_funds = (qty1 * symbol1_price) + (qty2 * symbol2_price * hedge_ratio)
            logging.info(f"Adjusted Required Funds for Trade: ${required_funds:.2f}")

            # Final check
            if required_funds > allocated_funds:
                logging.warning(f"Even after adjustment, required funds (${required_funds:.2f}) exceed allocated funds (${allocated_funds:.2f}). Skipping trade.")
                return

        # Proceed with trade based on signal
        if signal == 'long':
            # Check if already in position
            if open_positions[symbol1] or open_positions[symbol2]:
                logging.info("Already in a position. Skipping trade.")
                return

            # Execute Long symbol1 and Short symbol2
            api.submit_order(
                symbol=symbol1,
                qty=qty1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            api.submit_order(
                symbol=symbol2,
                qty=qty2,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"{current_time}: Executed LONG {symbol1} ({qty1} shares) and SHORT {symbol2} ({qty2} shares)")
            open_positions[symbol1] = 'long'
            open_positions[symbol2] = 'short'

        elif signal == 'short':
            # Check if already in position
            if open_positions[symbol1] or open_positions[symbol2]:
                logging.info("Already in a position. Skipping trade.")
                return

            # Execute Short symbol1 and Long symbol2
            api.submit_order(
                symbol=symbol1,
                qty=qty1,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            api.submit_order(
                symbol=symbol2,
                qty=qty2,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"{current_time}: Executed SHORT {symbol1} ({qty1} shares) and LONG {symbol2} ({qty2} shares)")
            open_positions[symbol1] = 'short'
            open_positions[symbol2] = 'long'

        elif signal == 'exit':
            # Check if positions are open
            if not open_positions[symbol1] and not open_positions[symbol2]:
                logging.info("No open positions to exit.")
                return

            # Close positions
            try:
                if open_positions[symbol1]:
                    api.close_position(symbol1)
                    logging.info(f"{current_time}: Closed position for {symbol1}")
                    open_positions[symbol1] = None
                if open_positions[symbol2]:
                    api.close_position(symbol2)
                    logging.info(f"{current_time}: Closed position for {symbol2}")
                    open_positions[symbol2] = None
            except Exception as e:
                logging.error(f"{current_time}: Error closing positions - {e}")

        else:
            # Hold position
            logging.info(f"{current_time}: No trading signal. Holding position.")

    except Exception as e:
        logging.error(f"Error in execute_trade: {e}")

def execute_pairs_trading():
    """
    Main function to execute pairs trading strategy.
    """
    try:
        # Fetch the latest data
        data1_latest = fetch_historical_data(symbol1, data_period, data_interval)
        data2_latest = fetch_historical_data(symbol2, data_period, data_interval)

        if data1_latest.empty or data2_latest.empty:
            logging.error("Failed to fetch latest data. Skipping this iteration.")
            return

        # Align data
        latest_combined = align_data(data1_latest, data2_latest)

        # Update combined_data
        global combined_data
        combined_data = latest_combined

        # Calculate hedge ratio
        beta = calculate_hedge_ratio(combined_data[f'{symbol1}_Close'], combined_data[f'{symbol2}_Close'])

        # Calculate spread and Z-score
        combined_data = calculate_spread_zscore(combined_data, beta)

        # Get the most recent timestamp
        current_time = combined_data.index[-1]

        # Get the latest Z-score
        latest_zscore = combined_data.loc[current_time, 'Zscore']
        logging.info(f"Current Z-score: {latest_zscore:.4f}")

        # Determine trading signal
        if latest_zscore > entry_zscore:
            signal = 'short'
        elif latest_zscore < -entry_zscore:
            signal = 'long'
        elif abs(latest_zscore) < exit_zscore:
            signal = 'exit'
        else:
            signal = 'hold'

        # Execute trade based on signal
        execute_trade(signal, current_time, hedge_ratio=beta, capital_allocation=capital_allocation)

    except Exception as e:
        logging.error(f"Error in execute_pairs_trading: {e}")

def run_trading_bot():
    """
    Schedules the trading bot to run during market hours at specified intervals.
    """
    # Initialize the bot by checking open orders and positions
    initialize_bot()

    # Define market open and close times
    today = datetime.now(tz).date()
    market_open = tz.localize(datetime.combine(today, datetime.strptime("09:30", "%H:%M").time()))
    market_close = tz.localize(datetime.combine(today, datetime.strptime("16:00", "%H:%M").time()))

    # Schedule the trading function to run every 5 minutes during market hours
    schedule.every(30).minutes.do(execute_pairs_trading)

    logging.info("Pairs Trading Bot is running...")

    while True:
        now = datetime.now(tz)
        if market_open <= now <= market_close:
            schedule.run_pending()
        else:
            logging.info(f"{now}: Market is closed. Waiting for the next trading session.")
        time.sleep(60)  # Check every minute

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_trading_bot()
