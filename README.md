## Pairs Trading Algorithm - NVDA vs TSM 💰🚀

This repository contains a Python script for a pairs trading algorithm that trades between Nvidia (NVDA) 🤖 and Taiwan Semiconductor Manufacturing Company (TSM) 🏭. 

**Disclaimer:** This is for educational purposes only and should not be considered financial advice. Backtesting results are not indicative of future performance.


### About Pairs Trading

Pairs trading is a market-neutral strategy that exploits temporary price discrepancies between two historically correlated assets. This algorithm identifies these discrepancies (spread) between NVDA and TSM and generates buy/sell signals when the spread deviates from its historical average.

**This script includes:**

* Data download (using a user-defined API)
* Preprocessing (cleaning, feature engineering)
* Cointegration test to identify statistically related pairs
* Spread calculation
* Trading signal generation based on spread deviation
* Backtesting on historical data

### Running the Algorithm

**Requirements:**

* Python 3.x
* Required libraries (listed in `requirements.txt`)

**Installation:**

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`
3. Update the `config.ini` file with your API key for data download (if applicable)

**Running Backtest:**

* Edit the `config.ini` file for parameters like lookback window, spread threshold, etc.
* Run the script: `python pairs_trading.py`

**Docker Support:**

This repository includes a `Dockerfile` to run the script in a Docker container.

1. Build the Docker image: `docker build -t pairs_trading .`
2. Run the backtest: `docker run pairs_trading python pairs_trading.py`

**Note:** These commands assume you have Docker installed and configured.

### Backtesting Results

The backtesting results are located in the `output` folder. This includes performance metrics like Sharpe Ratio, drawdown, and cumulative returns.

**Disclaimer:** Backtesting results are based on historical data and may not reflect future performance.


### Further Notes

* This is a basic implementation of a pairs trading algorithm. 
* You can customize the script with different features, risk management strategies, and more complex trading signals.
* Remember to conduct thorough research and understand the risks before deploying any trading strategy with real capital.


**License:**

This project is licensed under the MIT License (see [LICENSE](LICENSE) file)
