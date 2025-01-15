# Analyze results
#
# Desc:
# Analyse results such as training loss, training and testing error
#
# Author: Yue(Julien) Niu

import numpy as np


def calc_hist(results):
    """calculate histograms of results
    :param results: simulation result data
    """
    data = {}
    for key in results[0]:
        if 'Error' in key:
            data[key] = []
            
    for item in results:
        for key in item:
            if 'Error' in key:
                data[key].append(item[key])
                
    # convert to numpy array and calculate histograms
    for key in data:
        data[key] = np.array(data[key])
        cnt, bin = np.histogram(data[key], bins=30, density=True)
        
        print('\n{}:\n'.format(key))
        for bin_i, cnt_i in zip(bin, cnt):
            print('{:.3f}, {:.3f}'.format(bin_i, cnt_i))
