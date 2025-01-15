import numpy as np

from Utils.utils import generate_metrics_given_config, merge_metrics, save_evaluation, save_model, make_save_path
from Utils.data_utils import *
from Model.model_wrapper import *


def subset_split(X, y, train_percentage, kfold=False, independent=False):

    split_size = np.gcd(int(train_percentage * 100), 100)
    split_time = int(100 / split_size)

    X_size = X.shape[1]
    combine = np.hstack((X, y))

    if independent:
        for _ in range(split_time):
            np.random.shuffle(combine)
            split_array = np.array_split(combine, split_time)
            concat_list = [split_array[k] for k in range(len(split_array)) if k != 0]
            if np.gcd(int(train_percentage * 100), 100) + int(train_percentage * 100) == 100:
                train_data = np.concatenate(concat_list)
                validate_data = split_array[0]
            else:
                train_data = split_array[0]
                validate_data = np.concatenate(concat_list)
            yield train_data[:,:X_size], validate_data[:,:X_size], train_data[:,X_size:], validate_data[:,X_size:]
            if not kfold:
                break

    else:
        np.random.shuffle(combine)
        split_array = np.array_split(combine, split_time)
        for st in range(len(split_array)):
            concat_list = [split_array[k] for k in range(len(split_array)) if k != st]
            if np.gcd(int(train_percentage * 100), 100) + int(train_percentage * 100) == 100:
                train_data = np.concatenate(concat_list)
                validate_data = split_array[st]
            else:
                train_data = split_array[st]
                validate_data = np.concatenate(concat_list)
            yield train_data[:,:X_size], validate_data[:,:X_size], train_data[:,X_size:], validate_data[:,X_size:]

            if not kfold:
                break


class Model:
    def __init__(self, train_config, model, train_parameter, train_performance, test_parameter, test_performance, data_config, scaler):
        self.train_config = train_config
        self.model = model
        self.train_parameter = train_parameter
        self.train_performance = train_performance
        self.test_parameter = test_parameter
        self.test_performance = test_performance
        self.data_config = data_config
        self.scaler = scaler

    def train(self):
        train_result = self.model.fit(self.train_performance, self.train_parameter, self.test_performance, self.test_parameter)
        return train_result
    
    def eval(self, save_folder):

        test_parameter_prediction = self.model.predict(self.test_performance)
        train_parameter_prediction = self.model.predict(self.train_performance)

        inverse_test_parameter, inverse_test_performance = inverse_transform(test_parameter_prediction, self.test_performance, self.scaler)
        inverse_train_parameter, inverse_train_performance = inverse_transform(train_parameter_prediction, self.train_performance, self.scaler)

        save_evaluation(inverse_train_parameter, inverse_train_performance, 
                        inverse_test_parameter, inverse_test_performance, 
                        self.data_config, self.train_config, 
                        save_folder)

    def reset(self):
        self.model.reset()


class ModelPipeline:
    def __init__(self, parameter, performance, data_config, train_config, model):

        if np.any(performance == 0):
            raise ValueError("There is 0 in performance before scaling")

        new_parameter, new_performance, data_scaler = transform_data(parameter, performance)

        self.parameter = new_parameter
        self.performance = new_performance
        self.data_config = data_config
        self.train_config = train_config
        self.scaler = data_scaler

        if train_config["model_type"] == 0:
            self.model_wrapper = SklearnModelWrapper(model)
        else:
            self.model_wrapper = PytorchModelWrapper(model, train_config)

    def run(self):

        subset = self.train_config["subset"]
        metrics_dict = generate_metrics_given_config(self.train_config)
        save_folder = make_save_path(self.data_config.arguments["out"], self.train_config["model_name"])
        
        for index, percentage in enumerate(subset):
            print("Running with percentage {}".format(percentage))
            if percentage == 1 or percentage > 1:
                raise ValueError("Subset Percentage must smaller than 1")
            if np.gcd(int(percentage * 100), 100) + int(percentage * 100) != 100 \
                    and np.gcd(int(percentage * 100), 100) != int(percentage * 100):
                raise ValueError("Subset Percentage must be divisble")

            subset_metrics_dict = generate_metrics_given_config(self.train_config)
            count = 0
            for (parameter_train, parameter_test, performance_train, performance_test) in subset_split(self.parameter, self.performance, percentage, 
                                                                                                       self.train_config["kfold"],
                                                                                                       self.train_config["independent_kfold"]):
                count += 1
                print("Run with {} percentage of training data, Run number {}".format(percentage, count))
                kfold_metrics_dict = generate_metrics_given_config(self.train_config)

                new_train_parameter, new_train_performance = modify_data(parameter_train, performance_train, parameter_test, performance_test, train=True)
                new_test_parameter, new_test_performance = modify_data(parameter_train, performance_train, parameter_test, performance_test, train=False)

                eval_model = Model(self.train_config, self.model_wrapper,
                                                new_train_parameter, new_train_performance,
                                                new_test_parameter, new_test_performance, self.data_config, self.scaler)
                
                subset_kfold_metrics_dict = eval_model.train()
                eval_model.eval(save_folder)
                
                merge_metrics(kfold_metrics_dict, subset_kfold_metrics_dict)
                merge_metrics(subset_metrics_dict, kfold_metrics_dict)

                if self.train_config["save_model"]:
                    save_model(eval_model.model, eval_model.scaler, save_folder)
                    
                eval_model.reset()

            merge_metrics(metrics_dict, subset_metrics_dict)
        return metrics_dict