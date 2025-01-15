from Pipeline.dataset import *
import os

from Utils.utils import load_circuit, saveDictToTxt, make_plot_folder
from Model.models import *
from Utils.visualutils import plot_loss


def circuit_config_creator(circuit_name):

    config_path = os.path.join(os.path.join(os.getcwd(), "Config"), "Circuits")

    circuits = ["CSVA", "TSVA", "CVA", "LNA", "Mixer", "VCO", "PA", 
                "Transmitter", "Receiver"]
    circuit_mapping = dict()

    for circuit in circuits:
        circuit_mapping[circuit.lower()] = os.path.join(config_path, circuit + ".yaml")

    if circuit_name.lower() in circuit_mapping:
        circuit_definition_path = circuit_mapping[circuit_name.lower()]
    else:
        raise KeyError("The circuit you defined does not exist")

    circuit = load_circuit(circuit_definition_path)
    return circuit


def generate_model_given_config(model_config,num_params,num_perf):

    sklearn_model_mapping = {
        "RandomForest": RandomForest,
        "SupportVector": SupportVector,
        "KNeighbors": KNeighbors,
    }

    dl_model_mapping = {
        "MultiLayerPerceptron": Model500GELU,
        "Transformer": Transformer
    }

    if model_config["model"] in sklearn_model_mapping.keys():
        eval_model = sklearn_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 0
    
    elif model_config["model"] in dl_model_mapping.keys():
        model_config['input_count'] = num_perf
        model_config['output_count'] = num_params
        eval_model = dl_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 1
    
    else:
        raise KeyError("The model you defined does not exist")


def generate_visual_given_result(result, train_config, visual_config, pipeline_save_name, circuit):

    save_folder = make_plot_folder(pipeline_save_name)
    result_dict = dict()
    if train_config["loss_per_epoch"]:
        loss_plot_result = plot_loss(train_config, visual_config, result, save_folder, circuit)
        result_dict.update(loss_plot_result)

    return result_dict


def generate_circuit_status(parameter, performance, path):

    circuit_dict = dict()
    circuit_dict["num_parameter"] = parameter.shape[1]
    circuit_dict["num_performance"] = performance.shape[1]
    circuit_dict["data_size"] = performance.shape[0]

    saveDictToTxt(circuit_dict, path)