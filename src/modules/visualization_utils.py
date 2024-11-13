import matplotlib.pyplot as plt

def plot_predictions(df, train_size, y_pred, y_test):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[train_size + 60:], y_test, label="Actual")
    plt.plot(df.index[train_size + 60:], y_pred, label="LSTM Prediction")
    plt.legend()
    plt.show()

def plot_future_forecast(df, train_size, y_pred, future_predictions, future_dates, rmse):
    plt.figure(figsize=(12, 6))
    plt.plot(df['TSLA'], label="Historical Data")
    plt.plot(df.index[train_size + 60:], y_pred, label="Test Forecast")
    plt.plot(future_dates, future_predictions, label="Future Forecast")
    plt.fill_between(future_dates,
                     (future_predictions - rmse).flatten(),
                     (future_predictions + rmse).flatten(),
                     color="k", alpha=0.1, label="Confidence Interval")
    plt.legend()
    plt.show()