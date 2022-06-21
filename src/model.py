import scipy 
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as k
from scipy.special import expit
from scipy.stats import nbinom
from lifelines.statistics import logrank_test
from sklearn.metrics import mutual_info_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as k
import tensorflow_probability as tfp


class SurvivalNN(keras.Model):

    def __init__(self, ninput, hidden, **params):

        super(SurvivalNN, self).__init__(**params)
        self.ninput = ninput
        self.hidden = hidden
        self.metric_loss = keras.metrics.Mean(name="loss")
        
        # bulding network 
        self.layers_ = [layers.Dense(self.ninput, activation=tf.nn.relu)]
        for h in self.hidden:
            self.layers_.append(layers.Dense(h, activation=tf.nn.relu))
        self.layers_.append(layers.Dense(1, activation='linear'))

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        return x

    # loss decoretor
    def custom_loss(self, flags):
        # flag establish which instance is censored (1 = cencosred)
        flags = tf.expand_dims(flags, axis=1)

        # actual loss
        def negbin_loss(y_true, y_pred):
            dist = tfp.distributions.NegativeBinomial(total_count=1, logits=y_pred) 
            y_true = tf.expand_dims(y_true, axis=1)
            return - k.sum((1 - flags) *  k.log(1 - dist.cdf(y_true) + k.epsilon()) + flags * k.log(dist.prob(y_true) + k.epsilon())) 

        return negbin_loss

    @tf.function
    def train_step(self, data):

        x, y = data
        y, flags = y[:, 0], y[:, 1]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            negbin_loss = self.custom_loss(c)
            loss = negbin_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update main metrics
        self.metric_loss.update_state(loss)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.metric_loss]
        # return [self.metric_loss, self.metric_regret]


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def eval(y_true, p, n=1):
    dist = nbinom(n, p)
    y_pred = nbinom.rvs(n, p, size=len(p))

    print("========================")
    print("==> MUTUAL INFORMATION")
    print(mutual_info_score(y_true, y_pred))
    print("==> JENSEN SHANNON DISTANCE")
    print(jensen_shannon_distance(y_true, y_pred))
    print("==> LOGRANK TEST")
    res = logrank_test(y_true, y_pred)
    print("= p-value: " + str(res.p_value))
    print("= test statistic: " + str(res.test_statistic))
    print("========================")



    



