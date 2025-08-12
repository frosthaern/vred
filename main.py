import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import tensorflow as tf


class AddFeatures:
    def __init__(self, df):
        self.df = df
        self.features = []

    def add(self, feature, value):
        self.df[feature] = value
        self.features.append(feature)


def load_clean_dataframe():
    df = pd.read_csv("train/ETH.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df.interpolate(method="linear", inplace=True)

    return df


def add_features(df):
    bid_volume_sum = df[[f"bid_volume{i}" for i in range(1, 6)]].sum(axis=1)
    ask_volume_sum = df[[f"ask_volume{i}" for i in range(1, 6)]].sum(axis=1)
    near_liq_bid = sum(
        df[f"bid_volume{i}"] / (df["mid_price"] - df[f"bid_price{i}"])
        for i in range(1, 6)
    )
    near_liq_ask = sum(
        df[f"ask_volume{i}"] / (df[f"ask_price{i}"] - df["mid_price"])
        for i in range(1, 6)
    )
    feature_builder = AddFeatures(df)
    feature_builder.add("spread", df["ask_price1"] - df["bid_price1"])
    feature_builder.add("mid_price_change", df["mid_price"].diff())
    feature_builder.add(
        "imbalance",
        (bid_volume_sum - ask_volume_sum) / (bid_volume_sum + ask_volume_sum),
    )
    feature_builder.add(
        "bid_dwap",
        (
            df[[f"bid_price{i}" for i in range(1, 6)]].values
            * df[[f"bid_volume{i}" for i in range(1, 6)]]
        ).sum(axis=1),
    )
    feature_builder.add(
        "ask_dwap",
        (
            df[[f"ask_price{i}" for i in range(1, 6)]].values
            * df[[f"ask_volume{i}" for i in range(1, 6)]]
        ).sum(axis=1),
    )
    feature_builder.add("rolling_window", df["mid_price"].rolling(window=5).std())
    feature_builder.add("near_liquidity", near_liq_bid + near_liq_ask)
    feature_builder.add(
        "microprice",
        (df["ask_price1"] * df["bid_volume1"] + df["bid_price1"] * df["ask_volume1"])
        / (df["bid_volume1"] + df["ask_volume1"]),
    )
    feature_builder.add(
        "bid_slope", (df["bid_price5"] - df["bid_price1"]) / bid_volume_sum
    )
    feature_builder.add(
        "ask_slope", (df["ask_price5"] - df["ask_price1"]) / ask_volume_sum
    )
    feature_builder.add("delata_obi", df["imbalance"].diff())

    print(f"features_added = {feature_builder.features}")
    return (feature_builder.df, feature_builder.features)


def test_featureset(feature_set, df):
    for feature in feature_set:
        test_feature(feature, df)


def test_feature(feature, df):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    x = df[[feature]].values
    y = df["label"].values.reshape(-1, 1)

    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    x = scalerx.fit_transform(x)
    y = scalery.fit_transform(y)

    SEQ_LEN = 60
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=x, targets=y, sequence_length=SEQ_LEN, batch_size=32, shuffle=True
    )

    dataset_size = len(list(dataset))
    train_size = int(0.8 * dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(1),  # regression
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mean_squared_error",
    )

    model.fit(train_dataset, epochs=5)

    y_pred_scaled = model.predict(test_dataset)
    y_pred = scalery.inverse_transform(y_pred_scaled)

    y_true_scaled = np.concatenate([y for _, y in test_dataset], axis=0)
    y_true = scalery.inverse_transform(y_true_scaled)

    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))


def main():
    df = load_clean_dataframe()
    df, features = add_features(df)
    feature_set = set(features)
    test_featureset(feature_set, df)


if __name__ == "__main__":
    main()
