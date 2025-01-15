import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_method_comparison(compare_loss_train, compare_loss_test, label, subsets, visual_config, epochs, circuit, save_folder):
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    plot_loss_with_method_comparison(compare_loss_train, label, subsets, visual_config, epochs, circuit, 'Train', save_folder)
    plot_loss_with_method_comparison(compare_loss_test, label, subsets, visual_config, epochs, circuit, 'Test', save_folder)


def plot_loss(train_config, visual_config, result, save_folder, circuit):
    
    plt.clf()

    train_loss_dict = generate_loss_statistics(result["train_loss"])
    plot_loss_with_subset_comparison(train_loss_dict, visual_config, train_config, save_folder, "Train", circuit)
    test_loss_dict = generate_loss_statistics(result["validation_loss"])
    plot_loss_with_subset_comparison(test_loss_dict, visual_config, train_config, save_folder, "Test", circuit)

    result_dict = dict()
    result_dict["multi_train_loss"] = train_loss_dict["multi_loss"]
    result_dict["multi_test_loss"] = test_loss_dict["multi_loss"]
    result_dict["multi_train_loss_lower_bound"] = train_loss_dict["multi_loss_lower_bounds"]
    result_dict["multi_test_loss_lower_bound"] = test_loss_dict["multi_loss_lower_bounds"]
    result_dict["multi_train_loss_upper_bound"] = train_loss_dict["multi_loss_upper_bounds"]
    result_dict["multi_test_loss_upper_bound"] = test_loss_dict["multi_loss_upper_bounds"]

    return result_dict


def plot_loss_with_method_comparison(compare_loss, labels, subsets, visual_config, epochs, data_name, data_type, save_folder):

    color = visual_config["color"]
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure()
    for percentage_index in range(len(subsets)):
        plt.clf()
        ax = fig.add_subplot()
        percentage_loss_mean_cross_comparison = [i[percentage_index] for i in compare_loss["mean_list"]]
        percentage_loss_upper_bound_cross_comparison = [i[percentage_index] for i in compare_loss["upper_bound_list"]]
        percentage_loss_lower_bound_cross_comparison = [i[percentage_index] for i in compare_loss["lower_bound_list"]]

        for compared_item_index in range(len(percentage_loss_mean_cross_comparison)):
            ax.plot(np.arange(epochs), percentage_loss_mean_cross_comparison[compared_item_index], label=labels[compared_item_index],
                    color=color[compared_item_index])
            ax.fill_between(np.arange(epochs), percentage_loss_lower_bound_cross_comparison[compared_item_index],
                            percentage_loss_upper_bound_cross_comparison[compared_item_index], alpha=.3,
                            color=color[compared_item_index])

        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel(f'{data_type} Loss')
        plt.xlabel("Epochs")
        plt.title(f'{data_name}')

        image_save_path = os.path.join(save_folder, f'Subset-{subsets[percentage_index]}-{data_type}-loss.png')
        plt.savefig(image_save_path, dpi=250)


def plot_loss_with_subset_comparison(loss_dict, visual_config, train_config, save_folder, data_type, data_name):

    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure()
    ax = fig.add_subplot()
    num_subset = len(loss_dict["multi_loss"])
    color = visual_config["color"][:num_subset]
    subset = train_config["subset"]
    epochs = train_config["epochs"]

    for i in range(len(loss_dict["multi_loss"])):
        
        if data_type == 'Train':
            temp_label = "{:3.1f}% data".format(subset[i] * 100)
        else:
            temp_label = "{:3.1f}% data".format((1-subset[i]) * 100)

        ax.plot(np.arange(epochs), loss_dict["multi_loss"][i], label=temp_label, color=color[i], linewidth=3)
        ax.fill_between(np.arange(epochs), loss_dict["multi_loss_lower_bounds"][i], loss_dict["multi_loss_upper_bounds"][i], alpha=.3, color=color[i])

        ax.set_xlim([0, epochs + 1])
        ax.set_ylim([0, None])

        if len(subset) > 1:
            ax.legend()

        plt.ylabel(f'{data_type} Loss')
        plt.xlabel("Epochs")
        plt.title(f'{data_name}')
            
        image_save_path = os.path.join(save_folder, data_type + "-loss.png")
        plt.savefig(image_save_path, dpi=250)


def generate_loss_statistics(loss):

    loss_dict = {"multi_loss": [], "multi_loss_lower_bounds": [], "multi_loss_upper_bounds": []}

    for percentage_loss_index in range(len(loss)):
        temp_loss = [loss[percentage_loss_index][i][0] for i
                     in range(len(loss[percentage_loss_index]))]
        temp_loss_mean = np.average(temp_loss, axis=0)
        temp_loss_std = stats.sem(temp_loss, axis=0) if len(temp_loss) > 1 else [np.nan for i in range(len(temp_loss[0]))]

        loss_dict["multi_loss"].append(temp_loss_mean)
        loss_dict["multi_loss_lower_bounds"].append(temp_loss_mean - temp_loss_std)
        loss_dict["multi_loss_upper_bounds"].append(temp_loss_mean + temp_loss_std)

    return loss_dict