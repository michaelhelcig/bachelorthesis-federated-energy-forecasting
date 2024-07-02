from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback
from keras.layers import Input

import pandas as pd
import numpy as np

import src.shared.constants as constants

def get_model():
    optimizer = Adam(learning_rate=constants.learning_rate)  # Optimizer
    model = Sequential()
    model.add(Input(shape=(constants.sequence_length, len(constants.features))))
    model.add(LSTM(units=constants.lstm_units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=constants.loss_function)

    return model


def get_weigths_np(weights_list):
    return [np.array(w, dtype=np.float32) for w in weights_list]


def are_subsequent(dates):
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days != 1:
            return False
    return True

def dates_to_daystrings(dates):
    return [date.strftime('%Y-%m-%d') for date in dates]