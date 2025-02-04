import numpy as np
from matplotlib import pyplot as plt 
import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data_per_epoch(fig, x, y, title="Model", log_iters=10, **kwargs):
    """Отображение данных по эпохам"""
    fig.set_title(title)
    fig.plot(x[::log_iters], y[::log_iters], **kwargs)
    fig.axvline(x[-1], c="black", linestyle="--")
    fig.hlines(y=np.mean(y), xmin=x[0], xmax=x[-1], colors="red")


def plot_data(fig, xs, ys, labels, title="Model", **kwargs):
    """Отображение нескольких графиков на одной картинке"""
    fig.set_title(title)
    fig.set_xticks(xs[0])
    for x, y, label in zip(xs, ys, labels):
        fig.plot(x, y, label=label, **kwargs)
    fig.legend()


def plot_graphs_of_education(axs, model_title, train_metrics, test_metrics, 
                             log_iters_train, log_iters_valid):
    """Построение графиков обучения с кучей данных"""
    train_it_num, train_loss, train_acc, train_fscore = np.split(np.array(train_metrics).T, [1, 2, 3])
    test_it_num, test_loss, test_acc, test_fscore = np.split(np.array(test_metrics).T, [1, 2, 3])
    plot_data_per_epoch(axs[0, 0], train_it_num[0], train_loss[0],
                            f"{model_title} train losses", log_iters_train)
    plot_data_per_epoch(axs[0, 1], train_it_num[0], train_acc[0],
                        f"{model_title} train accuracy", log_iters_train)
    plot_data_per_epoch(axs[0, 2], train_it_num[0], train_fscore[0],
                    f"{model_title} train fscore", log_iters_train)
    plot_data_per_epoch(axs[1, 0], test_it_num[0], test_loss[0],
                        f"{model_title} valid losses", log_iters_valid)
    plot_data_per_epoch(axs[1, 1], test_it_num[0], test_acc[0],
                        f"{model_title} valid accuracy", log_iters_valid)
    plot_data_per_epoch(axs[1, 2], test_it_num[0], test_fscore[0],
                        f"{model_title} valid fscore", log_iters_valid)