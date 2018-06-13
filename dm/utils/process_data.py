#-*- coding: utf-8 -*-

import numpy as np
import os
import random
import copy
import operator
from collections import OrderedDict
from pprint import pprint

import utils.data_helpers as data_helpers

MAX_FEAT_DIM = 5000

class Data:

    def __init__(self, parameters):
        self.parameters         = parameters
        self.num_classes        = 0             # number of classes to be classified
        self.class_to_idx_dict  = {}            # key : class명, value : class에 대응하는 index값
        self.idx_to_class_dict  = {}            # key : class에 대응하는 index값, value : class명
        self.idx_to_docpath_dict = {}           # key / value = index / document 경로
        self.docpath_to_idx_dict = {}           # key / value = document 경로 / index

        self.Tf_Idf_Dict        = OrderedDict()
        self.vocab              = {}
        self.reverse_vocab      = {}

        self.train_idx          = []
        self.train_target_idx   = []
        self.valid_idx          = []
        self.valid_target_idx   = []
        self.test_idx           = []
        self.test_target_idx    = []

        self.TF_IDF_Feature_Matrix = []

        self.train_data         = []
        #self.valid_data         = []
        self.test_data          = []

        self.POS_Feature        = []        # construct_document_representation 수행 후 저장
        self.num_feat_dim       = 0
        self.check_flag         = 0

    # 핵심 메소드
    def data_preprocessing(self):
        if os.path.exists(self.parameters['dataset_dirname']) is True:
            pass
        else:
            print ("Input_Data 폴더명과 위치를 다시 확인하고 프로그램을 수행하세요")
            exit(-1)

        self.set_answer_info()
        self.set_docpath_info()

        TF_IDF_Feature_Matrix, target_idx_list = self.read_student_feature(self.parameters['dataset_dirname'])
        train_idx, train_target_idx = self.make_data(target_idx_list, TF_IDF_Feature_Matrix, False)

        # A list of tuples consisting of (target label, tf_idf_feature vector)
        self.train_data = data_helpers.batch_construction(train_target_idx, train_idx)

        print ("")
        print ("  >> Implementation1 is complete!!")
        print ("  >> Do the next task, Implementation2.")
        print ("")

    def read_student_feature(self, dirpath):
        std_file_path_list = []
        std_file_path_list = self.dir_scan(dirpath, std_file_path_list)
        std_file_path_list.sort()

        TF_IDF_Feature_Matrix = []
        target_idx_list = []

        for docpath in std_file_path_list:
            with open(docpath, "r") as f:
                target_idx_list.append(self.find_class_idx(docpath, self.class_to_idx_dict))
                tmp_feat = f.readline()
                tmp_feat = [float(i) for i in tmp_feat.strip().split()]
                tmp_feat = np.array(tmp_feat)

                # TF_IDF Feature Dimension check
                assert len(tmp_feat) == MAX_FEAT_DIM, \
                    'TF_IDF Feature Dimension does not match MAX_FEAT_DIM'

                TF_IDF_Feature_Matrix.append(tmp_feat)

        TF_IDF_Feature_Matrix = np.array(TF_IDF_Feature_Matrix)

        assert len(target_idx_list) == len(TF_IDF_Feature_Matrix), \
            '# of rows in TF_IDF_Feature_Matrix should be equal to size of target_idx_list'

        self.check_flag = self.check_normalization(TF_IDF_Feature_Matrix)

        return TF_IDF_Feature_Matrix, target_idx_list

    def check_normalization(self, TF_IDF_Feature_Matrix):
        check_list=[]
        for idx, feat_vector in enumerate(TF_IDF_Feature_Matrix):
            if idx == 10:
                break
            sum_vector = np.sqrt(np.sum(np.square(feat_vector)))
            #print (feat_vector)
            #print (sum_vector)
            #A = input()
            check_list.append(sum_vector)

        #print (check_list)

        flag = False
        for elem in check_list:
            if elem == 1:
                flag = True
                break

        if flag is True:
            return 1
        else:
            return 0

    def convert_dict_to_matrix(self, target_dict):
        num_row = len(target_dict)
        for elem in target_dict:
            num_col = len(target_dict[elem])
            break
        dict_to_list = []
        for doc_path in target_dict:
            dict_to_list.append(list(target_dict[doc_path].values()))

        new_matrix = np.asarray(dict_to_list)

        return new_matrix

    def set_answer_info(self):
        dir_name_list = []
        dir_list = os.listdir(self.parameters['dataset_dirname'])
        for elem in dir_list:
            if os.path.isdir(os.path.join(self.parameters['dataset_dirname'], elem)):
                dir_name_list.append(elem)
        dir_name_list.sort()
        #print (dir_name_list)
        #print ("PAUSE")
        #A = input()
        self.num_classes = len(dir_name_list)
        for idx, dir_name in enumerate(dir_name_list):
            #print (dir_name, idx)
            #A = input()
            self.class_to_idx_dict[dir_name] = idx
            self.idx_to_class_dict[idx] = dir_name

        ''' # Debug
        # 카테고리 Label의 index를 확인
        print (self.class_to_idx_dict)
        print (self.idx_to_class_dict)
        '''

    def set_docpath_info(self):
        docpath_list = []
        docpath_list = self.dir_scan(self.parameters['dataset_dirname'], docpath_list)
        docpath_list.sort()

        '''# Debug
        # docpath_list를 확인
        print (len(docpath_list))
        '''
        for idx, docpath in enumerate(docpath_list):
            self.idx_to_docpath_dict[idx] = docpath
            self.docpath_to_idx_dict[docpath] = idx


    def fetch_docpath(self, docpath_idx_list, idx2docpath_dict):
        docpath_list = []
        for doc_idx in docpath_idx_list:
            #print (doc_idx, idx2docpath_dict[doc_idx])
            docpath_list.append(idx2docpath_dict[doc_idx[1]])

        return docpath_list


    def make_data(self, target_idx_list, TF_IDF_Feat_Matrix, is_test=False):

        train_idx = []              # document index
        train_target_idx = []       # Class index for a document

        corpus_size = len(target_idx_list)
        shuffled_idx = np.arange(corpus_size)
        if is_test is False:
            np.random.shuffle(shuffled_idx)

        for idx, data_idx in enumerate(shuffled_idx):
            train_idx.append(TF_IDF_Feat_Matrix[data_idx])
            train_target_idx.append(target_idx_list[data_idx])

        return train_idx, train_target_idx

    def find_class_idx(self, doc_path, class2idx_dict):
        doc_path_list = doc_path.strip().split('/')
        return class2idx_dict[doc_path_list[1]]

    def dictionary_shuffle(self, dict_info):
        #tmp_items = list(dict_info.items())
        tmp_keys = list(dict_info.keys())
        random.shuffle(tmp_keys)
        #new_dict = OrderedDict(copy.deepcopy(tmp_items))
        new_dict = OrderedDict()
        for elem in tmp_keys:
            new_dict[elem] = copy.deepcopy(dict_info[elem])

        return new_dict

    def dir_scan(self, dirname, filescan_list):
        '''
            - 디렉토리 경로를 저장해서 리스트 형태로 리턴
        '''
        try:
            dir_list = os.listdir(dirname)
            for dir_elem in dir_list:
                dir_elem = os.path.join(dirname, dir_elem)

                if os.path.isdir(dir_elem):
                    self.dir_scan(dir_elem, filescan_list)
                else:
                    ext = os.path.splitext(dir_elem)[-1]
                    if ext == '.txt':
                        #print (dir_elem)
                        filescan_list.append(dir_elem)

        except PermissionError:
            pass

        return filescan_list

