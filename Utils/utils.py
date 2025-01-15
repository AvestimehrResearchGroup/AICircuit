import yaml
import os
from os.path import join
import numpy as np
from torch.cuda import is_available
import shutil
import pandas as pd
import torch
import pickle


CONFIG_PATH = join(os.path.join(os.getcwd(), "Config"))

DEFAULT_TRAIN_CONFIG_PATH = join(CONFIG_PATH, "train_config.yaml")
DEFAULT_VISUAL_CONFIG_PATH = join(CONFIG_PATH, "visual_config.yaml")

DEFAULT_RESULT_FOLDER_PATH = join(os.getcwd(), "out_result")
DEFAULT_PLOT_FOLDER_PATH = os.path.join(os.getcwd(), "out_plot")


def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        config_yaml = yaml.safe_load(file)
    return config_yaml


def load_circuit(circuit_name):
    """Load circuit"""
    config_folder = join("config", "circuits")
    folder = join(config_folder, circuit_name)
    print(folder)
    config = load_yaml(folder)

    return config


def load_train_config(configpath=DEFAULT_TRAIN_CONFIG_PATH):

    train_config = load_yaml(configpath)
    if train_config["device"] == "cuda":
        if is_available():
            train_config["device"] = "cuda:0"
        else:
            train_config["device"] = "cpu"

    return train_config


def load_visual_config(configpath=DEFAULT_VISUAL_CONFIG_PATH):
    return load_yaml(configpath)


def load_data(data_config, circuit):
    parameter_path = join(data_config.arguments["input"], "x.npy")
    performance_path = join(data_config.arguments["input"], "y.npy")

    if not os.path.exists(parameter_path) or not os.path.exists(performance_path):
        print("Create numpy files of data")
        csv_data_to_numpy(parameter_path, performance_path, data_config, circuit)

    parameter= np.load(parameter_path, allow_pickle=True)
    performance =np.load(performance_path, allow_pickle=True)

    return parameter, performance


def csv_data_to_numpy(parameter_path, performance_path, data_config, circuit):

    path = join(data_config.arguments["input"], f'{circuit}.csv')

    if not os.path.exists(path):
        raise KeyError("The dataset doesn't exist in the defined path")

    data = pd.read_csv(path)

    x = np.asarray(data.iloc[:,0:data_config.num_params])
    np.save(parameter_path, x)

    y = np.asarray(data.iloc[:,data_config.num_params:])
    np.save(performance_path, y)


def parsetxtToDict(file_path):
    with open(file_path, "r") as file:
        file_info = file.readlines()
        return_dict = dict()

        for line in file_info:
            line_info = line.strip().split(":")
            try:
                return_dict[line_info[0]] = float(line_info[1])
            except ValueError:
                return_dict[line_info[0]] = line_info[1]
        return return_dict
    

def generate_metrics_given_config(train_config):

    metrics_dict = dict()
    if train_config["loss_per_epoch"]:
        metrics_dict["train_loss"] = []
        metrics_dict["validation_loss"] = []

    return metrics_dict


def merge_metrics(parent_metrics, child_metrics):

    for k in parent_metrics.keys():
        if k in child_metrics.keys():
            parent_metrics[k].append(child_metrics[k])


def save_result(result, pipeline_save_name, config_path=None):

    save_folder = join(DEFAULT_RESULT_FOLDER_PATH, pipeline_save_name)
    os.makedirs(save_folder)
    for k in result.keys():
        out_variable_save_path = join(save_folder, k + ".npy")
        np.save(out_variable_save_path, result[k])

    if config_path is not None:
        shutil.copyfile(config_path, join(save_folder, "train_config.yaml"))


def save_numpy_results(array, save_name, save_folder):  

    path = join(save_folder, save_name)
    np.save(path, array)
    

def save_csv_results(x, y, save_name, data_config, save_folder):

    df = pd.DataFrame(columns=data_config.parameter_list + data_config.performance_list)

    for i in range(x.shape[0]):
        df.loc[i] = list(x[i]) + list(y[i])

    path = join(save_folder, save_name)
    df.to_csv(path, index=False)


def saveDictToTxt(dict, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        count = 0
        for k,v in dict.items():
            if count != 0:
                file.write('\n')
            file.write(str(k) + ":" + str(v))
            count += 1


def sortVector(parameter, performance):

    data = np.hstack((performance, parameter))

    for i in range(len(performance.shape)):
        data = sorted(data, key=lambda x: x[i], reverse=True)
    data = np.array(data)
    return data[:, performance.shape[1]:], data[:, :performance.shape[1]]


def checkAlias(parameter, performance):

    sort_parameter, sort_performance = sortVector(parameter, performance)

    counter = 0
    duplicate_amount = 0
    while counter <= len(sort_performance) - 2:
        if np.all(np.equal(sort_performance[counter], sort_performance[counter + 1])):
            print("BELOW ARE THE DUPLICATE CASE")
            print("THE TWO DIFFERENT PARAMETER")
            print(sort_parameter[counter])
            print(sort_parameter[counter + 1])
            print("THE SAME RESULT PERFORMANCE")
            print(sort_performance[counter])

            duplicate_amount += 1
        counter += 1

    print("TOTAL DUPLICATE CASE IS {}".format(duplicate_amount))
    if duplicate_amount > 0:
        raise ValueError("THERE ARE ALIASING IN THE RESULT")


def single_pipeline_train_config_creator(train_config, model_config):

    new_train_config = dict(train_config)

    del new_train_config["circuits"]
    del new_train_config["model_config"]

    if "extra_args" in model_config.keys():
        for k,v in model_config["extra_args"].items():
            new_train_config[k] = v

    return new_train_config


def update_train_config_given_model_type(model_type, train_config, single_model_config):

    train_config["loss_per_epoch"] = True if "loss_per_epoch" not in train_config else train_config["loss_per_epoch"]

    if model_type == 0:
        #Sklearn model, so no loss and accuracy per epochs
        train_config["loss_per_epoch"] = False
    else:
        #Pytorch model, so have loss per epoch
        train_config["epochs"] = 100 if "epochs" not in train_config else train_config["epochs"]

    train_config["model_type"] = model_type
    train_config["model_name"] = single_model_config["model"]


def check_comparison_value_diff(train_config, value, key):
    if value is None:
        if key in train_config.keys():
            return train_config[key]
        else:
            return None
    else:
        if key not in train_config.keys() or train_config[key] != value:
            raise ValueError("The {} across different comparison is not the same".format(key))
        else:
            return value
        

def make_plot_folder(pipeline_save_name):

    save_folder = join(DEFAULT_PLOT_FOLDER_PATH, pipeline_save_name)
    os.makedirs(save_folder)
    return save_folder


def save_model(model, scaler, save_folder):

    filename = join(save_folder, 'model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    filename = join(save_folder, 'scaler.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)


def make_save_path(data_name, model_name):

    save_folder = join(os.getcwd(), data_name, model_name)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def save_evaluation(inverse_train_parameter, inverse_train_performance, 
                    inverse_test_parameter, inverse_test_performance, 
                    data_config, train_config, 
                    save_folder):

        if train_config["save_format"] == "csv":
            save_csv_results(inverse_test_parameter, inverse_test_performance, "test.csv", data_config, save_folder)
            save_csv_results(inverse_train_parameter, inverse_train_performance, "train.csv", data_config, save_folder)
        elif train_config["save_format"] == "numpy":
            save_numpy_results(inverse_test_parameter, "test_x.npy", save_folder)
            save_numpy_results(inverse_test_performance, "test_y.npy", save_folder)
            save_numpy_results(inverse_train_parameter, "train_x.npy", save_folder)
            save_numpy_results(inverse_train_performance, "train_y.npy", save_folder)