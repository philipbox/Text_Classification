#-*- coding:utf-8 -*-
import model.train as  train
import model.test as test
import tensorflow as tf

if __name__ == '__main__':
    train.train()
    tf.reset_default_graph()    # Initialize graph
    test.test()
