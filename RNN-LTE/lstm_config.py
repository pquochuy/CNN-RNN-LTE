#import tensorflow as tf
import numpy as np
import os

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train):
        # Input data
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Trainging
        #self.learning_rate = 0.0025
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.0001
        self.training_epochs = 100
        self.batch_size = 100
        self.dropout_keep_prob = 0.9

        # LSTM structure
        self.n_layers = 2  # Number of LSTM layers
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 256  # nb of neurons inside the neural network
        self.n_classes = 19  # Final output classes

