import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf

def order_book_imbalance(df, levels=5, window=10):
    bid_vol = sum(df[f"bid_volume{i}"] for i in range(1, levels + 1))
    ask_vol = sum(df[f"ask_volume{i}"] for i in range(1, levels + 1))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol).rolling(window).mean()


def window_creation(df, feature, window_size=10):
    x = df[feature].values
    x_values = []

    for i in range(window_size, len(x)):
        x_values.append(x[i - window_size : i])
    return np.array(x_values)

def main():
    # just loading
    window_size = 10
    df = pd.read_csv("train/ETH.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df.interpolate(method="linear", inplace=True)

    # convert the features into a rolling window features, then you don't need to do that shape while training
    df["spread"] = ((df["ask_price1"] - df["bid_price1"]).abs() / df["mid_price"]).abs()
    df["spread_rolling"] = df["spread"].rolling(window_size).mean()
    df["returns"] = df["mid_price"].pct_change()
    df["realized_vol_10s"] = df["returns"].rolling(window_size).mean()
    df["order_imbalance"] = order_book_imbalance(df).rolling(window_size).mean()
    df["microprice"] = (
        df["bid_price1"] * df["ask_volume1"] + df["ask_price1"] * df["bid_volume1"]
    ) / (df["bid_volume1"] + df["ask_volume1"])
    df["microprice_rolling"] = df["microprice"].rolling(window_size).mean()

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    features = [
        "spread_rolling",
        "realized_vol_10s",
        "order_imbalance",
        "microprice_rolling",
    ]

    y = df["label"]
    xscaler = sk.preprocessing.StandardScaler()
    x = xscaler.fit_transform(df[features])

    x = window_creation(df, feature=features, window_size=window_size)

    x_lstm = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y_lstm = y[window_size:]

    xtrain, xtest, ytrain, ytest = sk.model_selection.train_test_split(
        x_lstm, y_lstm, test_size=0.2, random_state=42
    )

    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)

    # model training
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(x.shape[1], x.shape[2])))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(xtrain, ytrain, epochs=10, batch_size=64)

    ypred = model.predict(xtest)

    print(f"rmse = {np.sqrt(sk.metrics.mean_squared_error(ytest, ypred))}")
    print(f"mae = {sk.metrics.mean_absolute_error(ytest, ypred)}")

    print(ypred.shape)
    print(ytest.shape)


if __name__ == "__main__":
    main()
