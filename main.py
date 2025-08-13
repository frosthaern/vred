import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf


def window_creation(df, feature, window_size=10):
    x = df[feature].values
    x_values = []

    for i in range(window_size, len(x)):
        x_values.append(x[i - window_size : i])
    return np.array(x_values)


def main():
    window_size = 10
    df = pd.read_csv("train/ETH.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df.interpolate(method="linear", inplace=True)
    print(f"total is nas = {df.isna().sum()}")

    df["spread"] = ((df["ask_price1"] - df["bid_price1"]).abs() / df["mid_price"]).abs()

    x = df[["spread"]]
    y = df["label"]
    xscaler = sk.preprocessing.StandardScaler()
    x = xscaler.fit_transform(x)

    x = window_creation(df, feature="spread", window_size=window_size)

    x_lstm = x.reshape(x.shape[0], x.shape[1], 1)
    y_lstm = y[window_size:]

    xtrain, xtest, ytrain, ytest = sk.model_selection.train_test_split(
        x_lstm, y_lstm, test_size=0.2, random_state=42
    )

    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(x.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(xtrain, ytrain, epochs=10, batch_size=64)

    ypred = model.predict(xtest)

    print(ypred.shape)
    print(ytest.shape)


if __name__ == "__main__":
    main()
