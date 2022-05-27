import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as k

# loss
def negbin_likelihood(y_true, y_pred):
    # y_true = RUL, y_pred = probability of going on
    dist = tfp.distributions.NegativeBinomial(total_count=1, logits=y_pred)
    return -k.sum(dist.log_prob(y_true))


# Function to build a regressor
def build_regressor(hidden):
    input_shape = (len(dt_in), )
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    model_out = layers.Dense(1, activation='linear')(x)
    model = keras.Model(model_in, model_out)
    return model
