# preprocessing_eda.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

def clean_data(df):
    """
    Clean missing values and fill them using forward fill method.
    """
    data=df.isna().sum()
    for col in df.columns:
        if data[col].dtype in([float,int]):
            if data[col] != 0:
                df[col]=df[col].fillna(df[col].mean())
        else:
            if data[col] != 0:
                df[col]=df[col].fillna(df[col].mode()[0])

    return df

def normalize_data(df):
    """
    Normalize data using MinMaxScaler for LSTM compatibility.
    """
    scaler = MinMaxScaler()
    numeric_columns=df.select_dtypes(include=[int,float]).columns
    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )
    return df

def plot_closing_prices(assets):
    """
    Plot the closing prices for the assets over time.
    """
    plt.figure(figsize=(14, 7))
    for name, data in assets.items():
        data['Close'].plot(label=name)
    plt.legend()
    plt.title("Closing Prices Over Time")
    plt.show()

def add_daily_change(df,df_name):
    """
    Calculate daily percentage change for each asset.
    """
    df['Daily Change'] = df['Close'].pct_change()
    df['Daily Change'][0]=df['Daily Change'].iloc[0] = 0 
    df["Daily Change"].plot(figsize=(10, 6))
    plt.title(f"Daily Percentage Change of {df_name}")
    plt.ylabel('Percentage Change')
    plt.xlabel('Date')
    plt.show()
    return df

def add_volatility(df,data_name,window=30):
    """
    Calculate rolling volatility (standard deviation) for daily returns.
    """
    df['Volatility'] = df['Daily Change'].rolling(window=window).std()
    rolling_mean = df['Daily Change'].rolling(window=30).mean()
    # rolling_std = df['Daily Change'].rolling(window=30).std()

    # Plot rolling mean and std deviation
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_mean, label='30-Day Rolling Mean')
    plt.plot(df["Volatility"], label='30-Day Rolling Std Dev', alpha=0.7)
    plt.title(f'30-Day Rolling Mean and Std Dev of Daily Returns of {data_name}')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    return df

def outlier_remove(df):
    z_scores = (df["Daily Change"] - df["Daily Change"].mean()) / df["Daily Change"].std()

    # Define a threshold for outliers (e.g., 3 standard deviations)
    outliers = z_scores[abs(z_scores) > 3]

    # Plot outliers
    plt.figure(figsize=(10, 6))
    plt.plot(df["Daily Change"].index,df["Daily Change"], label='Daily Percentage Change')
    plt.scatter(outliers.index, outliers, color='red', label='Outliers')
    plt.title('Outliers in Daily Percentage Change')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.show()

    return z_scores
def Seasonality_and_trend(df):
    tsla_close = df['Close']
    decomposition = sm.tsa.seasonal_decompose(tsla_close, model='multiplicative', period=252)

    # Plot the decomposition
    decomposition.plot()
    plt.show()