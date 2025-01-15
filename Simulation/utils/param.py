# Get circuit parameters
#
# Desc:
# This file contains functions that generate circuit parameters such as
# - circuit topology
# - circuit path
# - dataset path
#
# Author: Yue(Julien) Niu

import yaml
from yaml.loader import SafeLoader

with open('./Config/sim_config.yml', 'r') as f:
    config = list(yaml.load_all(f, Loader=SafeLoader))


def get_circ_params(circuit: str):
    circuit_params = None

    for item in config:
        if circuit == item['circuit']:
            return item['parameters']

    if not circuit_params:
        raise NotImplementedError


def get_circ_path(circuit: str):
    """
    locate the Ocean script path given the circuit
    :param circuit: circuit type
    :return: full Ocean script path, e.g. /path/to/Ocean/circuit
    """
    circuit_path = None
    circuit_path_docker = None

    for item in config:
        if circuit == item['circuit']:
            return item['ocean'] + '/' + circuit, item['oceandocker'] + '/' + circuit

    if not circuit_path or not circuit_path_docker:
        raise NotImplementedError


def get_dataset_path(circuit: str, model: str, train=False):
    """
    locate dataset path given the circuit and the model type
    :param circuit: circuit type
    :param model: model type
    :param train: train or test dataset
    :return: full dataset path, e.g. /path/model/circuit
    """
    dataset_path = None

    for item in config:
        if circuit == item['circuit']:
            if train:
                return item['dataset'] + '/' + circuit + '/' + model + '/train.csv'
            else:
                return item['dataset'] + '/' + circuit + '/' + model + '/test.csv'

    if not dataset_path:
        raise NotImplementedError
