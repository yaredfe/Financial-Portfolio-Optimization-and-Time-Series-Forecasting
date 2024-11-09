# Financial-Portfolio-Optimization-and-Time-Series-Forecasting
This project applies time series forecasting models (ARIMA, SARIMA, LSTM) on financial data from YFinance for assets like TSLA, BND, and SPY. The aim is to predict market trends, optimize portfolio allocation, and maximize returns while minimizing risks. Key tasks include data preprocessing, model development, and portfolio optimization analysis.

## step 1: Preprocess and Explore the Data

This task involves loading, cleaning, and exploring historical financial data for **TSLA** (Tesla), **BND** (Bonds), and **SPY** (S&P 500 ETF) to prepare it for modeling. The steps include data extraction, data cleaning, exploratory data analysis (EDA), and calculating key risk metrics.

### tasks

1. **Load Data Using YFinance**  
   - Extract historical data for **TSLA**, **BND**, and **SPY** using the `yfinance` library.
   - Define start and end dates to retrieve data for the specified period.

2. **Data Cleaning and Understanding**
   - **Check for Missing Values**: Identify any missing data and handle it using forward-fill or interpolation.
   - **Data Type Verification**: Ensure all columns (e.g., `Close`, `Open`, `High`, `Low`, `Volume`) are in numeric format.
   - **Basic Statistics**: Use summary statistics to understand data distribution.

3. **Normalize or Scale Data (if required)**
   - Normalize the closing prices using `MinMaxScaler` for models sensitive to scaling.

4. **Exploratory Data Analysis (EDA)**
   - **Visualize Closing Prices**: Plot the closing prices of **TSLA**, **BND**, and **SPY** to observe long-term trends.
   - **Daily Percentage Change**: Calculate and visualize daily returns to understand short-term volatility.
   - **Rolling Mean and Standard Deviation**: Calculate 30-day rolling statistics to analyze trends and volatility.
   - **Outlier Detection**: Identify significant anomalies using Z-scores or threshold-based methods.

5. **Seasonality and Trend Analysis**
   - Decompose the time series to analyze trend, seasonality, and residuals using `statsmodels`.

6. **Volatility Analysis**
   - Calculate rolling means and standard deviations to assess short-term volatility of daily returns.

7. **Key Insights and Risk Metrics**
   - **Value at Risk (VaR)**: Calculate the potential loss in value at a 99% confidence level.
   - **Sharpe Ratio**: Evaluate the risk-adjusted return of each asset by calculating the ratio of average return to the standard deviation of returns.