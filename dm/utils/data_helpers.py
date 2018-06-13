#-*- coding: utf-8 -*-

import numpy as np

def batch_construction(target_list, data_list):
    batch_list = []
    for target_idx, data_elem in zip(target_list, data_list):
        tmp_tuple = (target_idx, data_elem)
        batch_list.append(tmp_tuple)

    return batch_list

def batch_iter(parameters, train_data):
    minibatch_size = parameters['batch_size']

    num_train_data = len(train_data)
    shuffled_sequence_idx = np.arange(num_train_data)
    np.random.shuffle(shuffled_sequence_idx)

    num_minibatch_per_epoch = int((num_train_data-1) / minibatch_size)+1
    for mini_idx in range(num_minibatch_per_epoch):
        start_idx = mini_idx * minibatch_size
        end_idx = min( (mini_idx+1)*minibatch_size, num_train_data)

        if start_idx >= end_idx:    break
        yield shuffled_sequence_idx[start_idx:end_idx]


def get_minibatch(dataset, minibatch_seq, is_test = False):
    '''
    :param dataset: (정답, {key: tf-idf} 사전)의 리스트
    :param minibatch_seq: 미니배치 인덱스
    :return:
    '''
    mini_X = []
    mini_Y = []
    original_target = []

    for minibatch_idx in minibatch_seq:
        mini_Y.append(dataset[minibatch_idx][0])
        if is_test == True:
            original_target.append(dataset[minibatch_idx][0])
        mini_X_elem_list = []
        #for elem in dataset[minibatch_idx][1].values():
        for elem in dataset[minibatch_idx][1]:
            mini_X_elem_list.append(elem)
        mini_X.append(mini_X_elem_list)

    numpy_array = np.asarray(a=mini_X)
    numpy_Y = np.asarray(mini_Y)
    numpy_Y = numpy_Y[:,np.newaxis]

    if is_test == True:
        return numpy_array, numpy_Y, original_target

    return numpy_array, numpy_Y
