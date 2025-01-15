from sklearn.preprocessing import MinMaxScaler
import numpy as np


def transform_data(parameter, performance):
    data = np.hstack((np.copy(parameter), np.copy(performance)))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data[:, :parameter.shape[1]], scaled_data[:, parameter.shape[1]:], scaler


def inverse_transform(parameter, performance, scaler):
    data = np.hstack((parameter, performance))
    data = scaler.inverse_transform(data)
    return data[:, :parameter.shape[1]], data[:, parameter.shape[1]:]


def modify_data(train_parameter, train_performance, test_parameter, test_performance, train=True):
    if train:
        return train_parameter, train_performance
    else:
        return test_parameter, test_performance