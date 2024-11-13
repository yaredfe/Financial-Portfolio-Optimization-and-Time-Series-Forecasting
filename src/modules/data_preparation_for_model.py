import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data(file_path,columns):
    df = pd.read_csv(file_path, parse_dates=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data.iloc[i:(i + time_step)])
        y.append(data.iloc[i + time_step])
        
    return np.array(X), np.array(y)

def prepare_train_test_data(df, time_step=60, train_size_ratio=0.8):
    train_size = int(len(df) * train_size_ratio)
    train_data, test_data = df[:train_size], df[train_size:]
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    return X_train, y_train, X_test, y_test