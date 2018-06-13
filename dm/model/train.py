#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle

import utils.data_helpers as data_helpers
import model.MLP as MLP

def load_preprocessing():
    with open("Pickle/parameters.bin", "rb") as f:
        parameters = pickle.load(f)

    with open("Pickle/data_info.bin", "rb") as f:
        data_info = pickle.load(f)

    return parameters, data_info

def create_model(session, parameters, data_info):

    Model = MLP.MLP(parameters=parameters, data_info=data_info)
    session.run(tf.global_variables_initializer())

    return Model

def train():
    print("  >> Loading preprocessing information...", "\n")
    parameters, data_info = load_preprocessing()

    print ("  >> Loading Train Data...", "\n")
    train_data = data_info.train_data

    train_loss_history = []
    train_acc_history = []

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config = session_conf) as sess:
        Model = create_model(sess,parameters, data_info)

        if Model.global_epoch_step.eval() + 1 > parameters['n_epoch']:
            print ("  >> Current Epoch: {}, Max Epoch: {}".format(Model.global_epoch_step.eval(), parameters['n_epoch']))
            print ("  >> End of Training....")
            exit(-1)

        for epoch_idx in range(parameters['n_epoch']):
            try:
                batches = data_helpers.batch_iter(parameters, train_data)
                for minibatch in batches:
                    #print (minibatch)
                    input_indices, target_indices = data_helpers.get_minibatch(\
                            dataset = train_data,\
                            minibatch_seq = minibatch)

                    feed_dict = {
                        Model.X     :   input_indices,\
                        Model.Y     :   target_indices}

                    #   Training model......
                    _, global_step, minibatch_loss, minibatch_accuracy = sess.run(\
                        [Model.train_op, Model.global_step, Model.loss, Model.accuracy], feed_dict)


                    #   Check Training Process
                    if (global_step+1) % parameters['evaluation_every'] == 0 :


                        # 매 "evaluation_every" step마다 train의 결과를 저장!!!
                        train_loss_history.append(minibatch_loss)
                        train_acc_history.append(minibatch_accuracy)

                        print ("  >> Global_Step # {} at {}-epoch".format(global_step, Model.global_epoch_step.eval()))
                        print ("        - Train Loss    : {:,.2f}".format(minibatch_loss))
                        print ("        - Train Accuracy: {:,.2f}".format(minibatch_accuracy))
                        print ("")

                Model.global_epoch_step_op.eval()   #Increment Global_epoch_step

            except KeyboardInterrupt:
                print("     - Training Process Terminated....")
                exit(-1)

        print ("  >> Save the last model...")
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(parameters['save_dir'], 'MLP.ckpt')
        saver.save(sess, checkpoint_path, global_step = global_step)

    print ("  >> End of Training...")
    print ("")
    print ("")


if __name__ == '__main__':
    train()