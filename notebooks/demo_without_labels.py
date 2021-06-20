#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import texthero as hero
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

__all__ = [hero]

sys.path.append('../')

struct_log = '../data/OutputDATA/aggregatedData.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file
files = glob.glob("../data/DATA/*.csv")
pathToCsvs = '../data/DATA'

if __name__ == '__main__':

    # download the model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # aggregate the data from multiple cvs and load them to the df
    df = pd.DataFrame()
    for f in files:
        csv = pd.read_csv(f)
        df = df.append(csv)
        os.unlink(f)
        df.to_csv(struct_log, encoding='utf-8')
    # shutil.rmtree(pathToCsvs)
    # os.mkdir(pathToCsvs)

    df = pd.read_csv(
        struct_log, parse_dates=['created_at'], index_col="created_at"
    )
    df.fillna(value='', inplace=True)

    # split df to df_train and df_test
    df_train, df_test = train_test_split(df, test_size=0.2)

    print(df_train.head())
    print(df_test.head())
    print(df_train.shape, df_test.shape)

    train_embeddings = embed(df_train['text'])
    # create list from np arrays
    use_to_train = np.array(train_embeddings)
    # add lists as dataframe column
    df_train['use_to_train'] = [v for v in use_to_train]

    test_embeddings = embed(df_test['text'])
    # create list from np arrays
    use_to_test = np.array(test_embeddings)
    # add lists as dataframe column
    df_test['use_to_test'] = [v for v in use_to_test]

    train_size = int(len(df_train))

    test_size = int(len(df_test))

    train = df_train.iloc[0:train_size]
    test = df_test.iloc[0:test_size]

    print(train.shape, test.shape)

    # scaler = StandardScaler()
    # scaler = scaler.fit(train[['use_to_train']])
    # df_train['use_to_train'] = scaler.transform(train[['use_to_train']])
    # df_test['use_to_test'] = scaler.transform(test[['use_to_test']])

    print(df_train.head())
    print(df_test.head())
    print(df_train.shape, df_test.shape)


    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)


    TIME_STEPS = 30
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(
        df_train.T.T, train[['use_to_train']], TIME_STEPS
    )

    X_test, y_test = create_dataset(
        df_test.T.T, test[['use_to_test']], TIME_STEPS
    )
    # couldn't add timestamps
    print(X_train.shape, X_test.shape)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=X_train.shape[2])
        )
    )
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    print(model.summary())

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )

    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    THRESHOLD = 0.65
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = test[TIME_STEPS:].close
    anomalies = test_score_df[test_score_df.anomaly == True]
