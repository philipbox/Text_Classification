#-*- coding: utf-8 -*-
import pickle
import os

import utils.parameter as parameter
import utils.process_data as process_data

def preprocessing():
    print("  >> It will take about one minute. please wait.......")
    parameters = parameter_setting()

    res1, res2 = dataset_setting(parameters)

    print ("  >> Resul1: {}".format(res1))
    print ("  >> Resul2: {}".format(res2))


def parameter_setting():
    parameters, conf_parameters = parameter.load_parameter()
    make_pickle('parameters.bin', parameters)

    return parameters

def dataset_setting(parameters):
    data_info = process_data.Data(parameters)
    data_info.data_preprocessing()
    make_pickle("data_info.bin", data_info)

    res1 = len(data_info.train_data)
    res2 = data_info.check_flag
    return res1, res2

def make_pickle(filename, info):
    dir_path = os.path.join('Pickle', filename)
    with open(dir_path, "wb") as f:
        pickle.dump(info, f)


if __name__ == '__main__':
    preprocessing()