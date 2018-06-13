import os
import configparser

def convert_configparser_to_dict(config):
    config_parser_dict = {s: dict(config.items(s)) for s in config.sections()}
    return config_parser_dict

def load_parameter(parameter_path = os.path.join('.', 'parameters.ini')):
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameter_path)
    nested_parameters = convert_configparser_to_dict(conf_parameters)

    parameters = {}
    for s, k in nested_parameters.items():
        parameters.update(k)
    for k, v in parameters.items():
        if k in ['max_feature_dim','max_ner_feature_dim','max_dp_feature_dim', 'n_epoch', 'batch_size', 'evaluation_every',\
                 'patience', 'hidden_layer_1_size', 'hidden_layer_2_size', 'pos_feature', 'ner_feature', 'dp_feature', 'data_class_new']:
            parameters[k] = int(v)

        if k in ['train_ratio', 'valid_ratio', 'test_ratio', 'learning_rate']:
            parameters[k] = float(v)

    return parameters, conf_parameters



