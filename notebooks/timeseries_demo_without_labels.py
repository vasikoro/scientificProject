#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

__all__ = [hub]

sys.path.append('../')

struct_log = '../data/OutputDATA/aggregatedData.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file
files = glob.glob("../data/DATA/*.csv")
pathToCsvs = '../data/DATA'
epsilon = 0.5  # threshold for estimating invariant space

if __name__ == '__main__':

    df = pd.DataFrame()
    for f in files:
        csv = pd.read_csv(f)
        df = df.append(csv)
        os.unlink(f)
        df.to_csv(struct_log, encoding='utf-8')
    # shutil.rmtree(pathToCsvs)
    # os.mkdir(pathToCsvs)

    df_train = pd.read_csv(
        struct_log, parse_dates=True, index_col="created_at"
    )

    df_test = pd.read_csv(
        struct_log, parse_dates=True, index_col="created_at"
    )

    print(df_train.head(1))

    print(df_test.head(1))

    fig, ax = plt.subplots()
    df_train.plot(legend=False, ax=ax)
    plt.show()

    fig, ax = plt.subplots()
    df_test.plot(legend=False, ax=ax)
    plt.show()

    # download the model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # generate embeddings
    train_embeddings = embed(df_train['text'])
    # create list from np arrays
    use_to_train = []
    use_to_train.extend(np.array(train_embeddings))
    # add lists as dataframe column
    df_train['use_to_train'] = [v for v in use_to_train]

    # test_embeddings = embed(df_test['text'])  # create list from np arrays
    # use_to_test = np.array(train_embeddings).tolist()  # add lists as dataframe column
    # df['use_to_test'] = [v for v in use_to_test]

    # Normalize and save the mean and std we get,
    # for normalizing test data.
    training_mean = df_train['use_to_train'].mean()
    # training_std = df_train['use_to_train'].std()
    try:

        df_training_value = (df_train['use_to_train'] - training_mean)
        # df_training_value = (df_train['use_to_train'] - training_mean).div(training_std)
    except ZeroDivisionError:
        df_training_value = 0
        print("---------Unknown Exception Occurred!!!-----------")
    print("Number of training samples:", len(df_training_value))
    print(df_training_value.values)

    TIME_STEPS = 20


    # Generated training sequences for use in the model.
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i: (i + time_steps)])
        return np.stack(output)


    x_train = create_sequences(df_training_value.values[0])
    print("Training input shape: ", x_train.shape)

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[0], x_train.shape[1])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    # Checking how the first sequence is learnt
    plt.plot(x_train[0])
    plt.plot(x_train_pred[0])
    plt.show()

    df_test_value = (df_test - training_mean) / training_std
    fig, ax = plt.subplots()
    df_test_value.plot(legend=False, ax=ax)
    plt.show()

    # Create sequences from test values.
    x_test = create_sequences(df_test_value.values)
    print("Test input shape: ", x_test.shape)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    plt.hist(test_mae_loss, bins=50)
    plt.xlabel("test MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
            anomalous_data_indices.append(data_idx)

    df_subset = df_test.iloc[anomalous_data_indices]
    fig, ax = plt.subplots()
    df_test.plot(legend=False, ax=ax)
    df_subset.plot(legend=False, ax=ax, color="r")
    plt.show()
