import tensorflow as tf
import numpy as np
#from sklearn import metrics
import os
from lstm_config import Config

class ASCLSTM(object):
    """
        A CNN for audio event classification.
        Uses a convolutional, max-pooling and softmax layer.
        """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
        self.Y = tf.placeholder(tf.float32, [None, config.n_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)


        # LSTM Network
        # Exchange dim 1 and dim 0
        X = tf.transpose(self.X, [1, 0, 2])
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        X = tf.reshape(X, [-1, config.n_inputs])
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]

        self.W = {
            'hidden': tf.Variable(tf.random_normal([config.n_inputs, config.n_hidden])),
            'output': tf.Variable(tf.random_normal([config.n_hidden, config.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([config.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([config.n_classes]))
        }


        # Linear activation, reshaping inputs to the LSTM's number of hidden:
        hidden = tf.nn.relu(tf.matmul(
            X, self.W['hidden']
        ) + self.biases['hidden'])
        # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        hidden = tf.split(0, config.n_steps, hidden)
        # New hidden's shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

        # Define LSTM cell of first hidden layer:
        lstm_cell = tf.nn.rnn_cell.GRUCell(config.n_hidden)

        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout_keep_prob) # input dropout

        # Stack two LSTM layers, both layers has the same shape
        lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.n_layers,state_is_tuple=True) # due to built with LN LSTM

        lsmt_layers = tf.nn.rnn_cell.DropoutWrapper(lsmt_layers, output_keep_prob=self.dropout_keep_prob) # output dropout

        with tf.name_scope("output"):
            # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
            outputs, final_state = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
            self.lstm_last_output = outputs[-1]

            self.score = tf.matmul(self.lstm_last_output, self.W['output']) + self.biases['output']
            self.pred_Y = tf.argmax(self.score, 1);

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # Loss,optimizer,evaluation
            l2_loss = config.l2_reg_lambda * \
                 sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            losses = tf.nn.softmax_cross_entropy_with_logits(self.score, self.Y)
            self.loss = tf.reduce_mean(losses) + l2_loss

        # Accuracy
        # with tf.device('/cpu:0'), tf.name_scope("accuracy"):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.pred_Y, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

