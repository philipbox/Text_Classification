import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle

import utils.data_helpers as data_helpers
import model.MLP as MLP

from collections import OrderedDict

def load_preprocessing():
    with open("Pickle/parameters.bin", "rb") as f:
        parameters = pickle.load(f)

    with open("Pickle/data_info.bin", "rb") as f:
        data_info = pickle.load(f)


    return parameters, data_info

def load_model(session, parameters, data_info):

    Model = MLP.MLP(parameters = parameters, data_info = data_info)

    saver = tf.train.Saver()
    #ckpt_path = os.path.join(parameters['save_dir'], parameters['model_name'])
    ckpt_path = parameters['save_dir']
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        #print (ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
        print ("  >> Parameters are restored from {}".format(ckpt.model_checkpoint_path))
    else:
        print ("  >> Failed to reload the model...")
        exit(-1)

    return Model

def test():
    print("  >> Loading preprocessing data...", "\n")
    parameters, data_info = load_preprocessing()

    print("  >> Loading Test Dataset...", "\n")
    TF_IDF_Feature_Matrix, target_idx_list = data_info.read_student_feature(parameters['dataset_testset'])
    test_idx, test_target_idx = data_info.make_data(target_idx_list, TF_IDF_Feature_Matrix, is_test = True)
    test_data = data_helpers.batch_construction(test_target_idx, test_idx)

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config=session_conf) as sess:
        Model = load_model(sess, parameters, data_info)

        test_input_indices, test_target_indices, test_target_origin =\
            data_helpers.get_minibatch(dataset=test_data,\
                                       minibatch_seq=np.arange(len(test_data)),\
                                       is_test = True)

        feed_dict = {
            Model.X: test_input_indices,\
            Model.Y: test_target_indices
        }

        test_logits = sess.run([Model.softmax_output], feed_dict=feed_dict)

        # Save the output of softmax layer
        np.savetxt(fname=parameters['output_path'], X=test_logits[0], \
                   fmt='%.10f', delimiter = '\t')

        np.savetxt(fname='answer.txt', X=test_target_indices, fmt='%d')

        print("  >> End of Test...")
        print("  >> Check 'output.txt' and 'answer.txt' file...")
        print("")



if __name__ == '__main__':
    test()