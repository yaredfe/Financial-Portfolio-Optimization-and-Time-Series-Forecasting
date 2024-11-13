import numpy as np

def make_predictions(model, X_test, scaler):
    y_pred = model.predict(X_test)
    # y_pred = np.repeat(y_pred, 8, axis=1)
    return y_pred

def forecast_future(model, last_sequence, future_steps=180, scaler=None):
    future_predictions = []
    for _ in range(future_steps):
        next_pred = model.predict(last_sequence.reshape(1, -1, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)  # Slide the window
    # future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions