import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf

def order_book_imbalance(df, levels=5, window=10):
    bid_vol = sum(df[f"bid_volume{i}"] for i in range(1, levels + 1))
    ask_vol = sum(df[f"ask_volume{i}"] for i in range(1, levels + 1))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol).rolling(window).mean()


def main():
    # just loading
    window_size = 10
    df = pd.read_csv("train/ETH.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df.interpolate(method="linear", inplace=True)

    # convert the features into a rolling window features, then you don't need to do that shape while training
    df["spread"] = (df["ask_price1"] - df["bid_price1"]).abs() / df["mid_price"]
    df["spread_rolling"] = df["spread"].rolling(window_size).mean()
    df["returns"] = df["mid_price"].pct_change(window_size)
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
    ]

    y = df["label"]

    # Scale features
    xscaler = sk.preprocessing.StandardScaler()
    x = xscaler.fit_transform(df[features])

    xtrain, xtest, ytrain, ytest = sk.model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)

    # model training
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(xtrain.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)

    ypred = model.predict(xtest)

    print(f"rmse = {np.sqrt(sk.metrics.mean_squared_error(ytest, ypred))}")
    print(f"mae = {sk.metrics.mean_absolute_error(ytest, ypred)}")

    df_test = pd.read_csv("test/ETH.csv")
    df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
    df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
    df_test.interpolate(method="linear", inplace=True)

    # convert the features into a rolling window features, then you don't need to do that shape while training
    df_test["spread"] = (df_test["ask_price1"] - df_test["bid_price1"]).abs() / df_test[
        "mid_price"
    ]
    df_test["spread_rolling"] = df_test["spread"].rolling(window_size).mean()
    df_test["returns"] = df_test["mid_price"].pct_change(window_size)
    df_test["realized_vol_10s"] = df_test["returns"].rolling(window_size).mean()
    df_test["order_imbalance"] = (
        order_book_imbalance(df_test).rolling(window_size).mean()
    )
    df_test["microprice"] = (
        df_test["bid_price1"] * df_test["ask_volume1"]
        + df_test["ask_price1"] * df_test["bid_volume1"]
    ) / (df_test["bid_volume1"] + df_test["ask_volume1"])
    df_test["microprice_rolling"] = df_test["microprice"].rolling(window_size).mean()

    df_test.ffill(inplace=True)
    df_test.bfill(inplace=True)

    x_test = xscaler.transform(df_test[features])

    ypred = model.predict(x_test)

    submission_df = pd.DataFrame(
        {
            "labels": ypred.flatten(),
        },
        index=pd.RangeIndex(start=1, stop=len(ypred) + 1, name="timestamp"),
    )

    submission_df.to_csv("submission.csv")


if __name__ == "__main__":
    main()
