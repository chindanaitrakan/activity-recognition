import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, concatenate
from keras.models import Sequential
import keras_tuner as kt


def dnn_model_builder(hp):

    model = Sequential()

    # Tune the number of units in the first and second Dense layer
    # Choose an optimal value between 8-128
    hp1_units = hp.Int('units1', min_value=8, max_value=128, step=8)
    hp2_units = hp.Int('units2', min_value=8, max_value=128, step=8)
    model.add(Dense(units=hp1_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=hp2_units, activation='relu'))
    hp_dropout1 = hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    model.add(Dropout(hp_dropout1))
    model.add(Dense(16))
    model.add(Dense(8, activation = 'softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    return model

def cnn_model_builder(hp):
    
    inputs = tf.keras.Input(shape=(64, 6, 1))

    hp_filter_width = hp.Int('width', min_value=1, max_value=4, step=1)
    hp_filter_height = hp.Int('height', min_value=3, max_value=18, step=1)

    conv2D_layer1 = Conv2D(32, (hp_filter_width, hp_filter_height), strides = (1,1), activation='relu', padding = 'same')(inputs)
    norm_layer1 = BatchNormalization()(conv2D_layer1)

    maxpooling_layer1 = MaxPooling2D(pool_size = (3, 2), strides = 1)(norm_layer1)
    conv2D_layer2 = Conv2D(16, (3,1), strides = (2,2), activation='relu', padding = 'same')(maxpooling_layer1)
    maxpooling_layer2 = MaxPooling2D(pool_size = (3, 2), strides = 1)(conv2D_layer2)
    flatten_layer = Flatten()(maxpooling_layer2)

    hp_units = hp.Int('units', min_value=8, max_value= 64, step=4)
    dense_layer1 = Dense(hp_units)(flatten_layer)
    dense_layer2 = Dense(16)(dense_layer1)
    dropout_layer1 = Dropout(0.7)(dense_layer2)
    outputs = Dense(8, activation = 'softmax')(dropout_layer1)

    model = tf.keras.Model(inputs, outputs)

    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

    return model

def lstm_model_builder(hp):
    
    inputs = tf.keras.Input(shape=(64, 6))

    hp1_units = hp.Int('units1', min_value=8, max_value=64, step=8)
    lstm_layer1, state_h, state_c= LSTM(hp1_units, return_sequences=True, return_state = True)(inputs)

    hp2_units = hp.Int('units2', min_value=8, max_value=32, step=8)
    lstm_layer2 = LSTM(hp2_units, return_sequences=True)(lstm_layer1)
    lstm_layer2_flatten = Flatten()(lstm_layer2)
    dense_layer1 = Dense(64)(concatenate([lstm_layer2_flatten, state_h, state_c]))
    dropout_layer1  = Dropout(0.6)(dense_layer1)
    dense_layer2 = Dense(16)(dropout_layer1)
    outputs = Dense(8, activation = 'softmax')(dense_layer2)

    model = tf.keras.Model(inputs, outputs)

    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3, 1e-3])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

    return model