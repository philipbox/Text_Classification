# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class MLP(object):
    def __init__(self, parameters, data_info, model_name='MLP'):
        self.parameters = parameters
        self.data_info = data_info
        self.feat_dim = parameters['max_feature_dim']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.build_graph()

    def build_graph(self):
        self.initialization()
        self.forward_propagation()
        self.calculate_loss()
        self.check_prediction_accuracy()

    def initialization(self):
        self.X = tf.placeholder( \
            dtype=tf.float32, \
            shape=(None, self.feat_dim), \
            name='input_data')

        self.Y = tf.placeholder( \
            dtype=tf.int32, \
            shape=(None, 1), \
            name='answer')


        #   tf.one_hot : rank가 1 증가함 [?, 1] --> [?, 1, num_class] --> reshape 필요 --> [?, num_class]
        self.Y_one_hot = tf.one_hot(self.Y, self.data_info.num_classes)
        self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, self.data_info.num_classes])

        # Input Layer -- Hidden Layer 1
        self.W1 = tf.get_variable(name='W1', \
                                  shape=[self.feat_dim, self.parameters['hidden_layer_1_size']], \
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b1 = tf.Variable(tf.random_normal([self.parameters['hidden_layer_1_size']]))

        # Hidden Layer1 -- Hidden Layer2
        self.W2 = tf.get_variable(name='W2', \
                                  shape=[self.parameters['hidden_layer_1_size'],
                                         self.parameters['hidden_layer_2_size']], \
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b2 = tf.Variable(tf.random_normal([self.parameters['hidden_layer_2_size']]))


        # Hidden Layer3 -- Output Layer
        self.W3 = tf.get_variable(name='W3', \
                                  # shape = [self.parameters['hidden_layer_2_size'], self.data_info.num_classes],\
                                  shape=[self.parameters['hidden_layer_2_size'], self.data_info.num_classes], \
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.Variable(tf.random_normal([self.data_info.num_classes]))

    def forward_propagation(self):
        self.L1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.W1), self.b1))
        self.L2 = tf.nn.relu(tf.add(tf.matmul(self.L1, self.W2), self.b2))
        # self.L3 = tf.nn.relu(tf.add(tf.matmul(self.L2, self.W4), self.b4))
        self.logits = tf.add(tf.matmul(self.L2, self.W3), self.b3)
        self.hypothesis = tf.nn.relu(self.logits)
        self.softmax_output = tf.nn.softmax(logits = self.logits)

    def calculate_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
            logits=self.logits, \
            labels=self.Y_one_hot))

        '''
            -   Increment by one after the variables have been updated.
            -   Global_step : the number of batches seen by the graph
            -   Everytime a batch is provided, the weights are updated in the direction 
                that minimizes the loss. 
                global_step just keeps track of the number of batches seen so far. 
                When it is passed in the minimize() argument list, the variable is increased by one.

        '''
        optimizer = tf.train.AdamOptimizer(self.parameters['learning_rate'])
        self.train_op = optimizer.minimize( \
            loss=self.loss, \
            global_step=self.global_step)

    def check_prediction_accuracy(self):
        # self.hypothesis : (batch_size * data_info.num_classes)
        #self.prediction = tf.argmax(input=self.hypothesis, axis=1)
        self.prediction = tf.argmax(input=self.softmax_output, axis=1)

        # tf.equal : True(x == y); False(x!=y)
        # self.prediction_correct : (batch_size * 1)
        self.prediction_correct = tf.equal( \
            x=self.prediction, \
            y=tf.argmax(self.Y_one_hot, 1))

        # tf.cast(A, tf.float32) : A의 datatype --> tf.float32로 치환
        # tf.reduce_mean(A) : A의 element의 평균값
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction_correct, tf.float32))

