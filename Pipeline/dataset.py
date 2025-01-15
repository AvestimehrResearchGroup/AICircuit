import numpy as np
from torch.utils.data import Dataset


def data_config_creator(circuit_config):

    return Data(**circuit_config)


class Data:
    def __init__(self, parameter_list, performance_list, arguments):

        self.arguments = dict(arguments)
        self.performance_list = list(performance_list)
        self.parameter_list = list(parameter_list)

        self.num_params = len(parameter_list)
        self.num_perf = len(performance_list)

        
class BasePytorchModelDataset(Dataset):
    def __init__(self, performance, parameters):
        self.parameters = np.array(parameters)
        self.performance = np.array(performance)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.performance[index], self.parameters[index]

    def getAll(self):
        return self.performance, self.parameters