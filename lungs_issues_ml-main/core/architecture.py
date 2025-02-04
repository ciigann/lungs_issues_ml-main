from tqdm import tqdm
import time
import os
from sklearn.metrics import accuracy_score, f1_score
import torch
from .graph_functions import plot_data, plot_graphs_of_education
from matplotlib import pyplot as plt
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# метрика для сегментации.
def intersection_over_union(predicted, target):
    # Переводим предсказанные вероятности в бинарный формат (0 или 1)
    predicted_binary = torch.round(predicted)
    
    # Вычисляем пересечение (intersection) и объединение (union) между предсказанными и истинными масками
    intersection = torch.logical_and(predicted_binary, target).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(predicted_binary, target).sum(dim=(1, 2, 3)).float()
    
    # Вычисляем значение IoU для каждого элемента в батче
    iou = intersection / union
    
    return iou

def mean_iou(predicted, target):
    # Вычисляем IoU для каждого элемента в батче
    iou = intersection_over_union(predicted, target)
    
    # Вычисляем среднее значение IoU для всего батча
    mean_iou = iou.mean()
    
    return mean_iou


def get_accuracy_fscore(output, labels):
    """Получение метрик модели: accuracy и fscore"""
    pred = output.argmax(1)
    return accuracy_score(labels, pred), f1_score(labels, pred, average="macro")


def go_for_epoch(data, batch_size, epoch_num, log_desc, model, loss_func, optimiser=None, segmentation=False):
    """Обучение/тест модели в течение одной эпохи"""
    if optimiser is not None:
        model.train()
    else:
        model.eval()
    ix = -1
    total_loss = 0
    total_acc = 0  # В задачах сегментации это mIoU.
    total_fscore = 0

    for x, y in tqdm(data, desc=log_desc):
        if len(y) != batch_size:
            continue
        x, y = x.to(device), y.to(device)
        if optimiser is not None:
            optimiser.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        if optimiser is not None:
            loss.backward()
            optimiser.step()
        ix += 1
        cur_loss = loss.item()
        if segmentation:
            cur_acc = mean_iou(y_pred.cpu(), y.cpu())
            cur_fscore = 0
        else:
            cur_acc, cur_fscore = get_accuracy_fscore(y_pred.cpu(), y.cpu())
        yield ix + len(data) * epoch_num, cur_loss, cur_acc, cur_fscore
        total_loss += cur_loss
        total_acc += cur_acc
        total_fscore += cur_fscore

    yield total_loss / len(data), total_acc / len(data), total_fscore / len(data)


def save_model_state(model, optimiser, model_title, epoch_num):
    """Сохранение состояния модели"""
    torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict()
            }, os.path.join("model_states", f"{model_title}.pt"))


def load_model_state(model_title, model, optimiser=None):
    """Загрузка состояния модели"""
    checkpoint = torch.load(os.path.join("model_states", f"{model_title}.pt"),
                            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimiser is not None:
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    return epoch


def train_model(dataset_train, dataset_test, model, optimiser, loss_func,
                num_epochs, batch_size, logging_iters_train,
                logging_iters_valid, model_title, save_graph, 
                save_state, load_state, period_save_weights,
                segmentation):
    model = model.to(device)
    data_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                             generator=torch.Generator(device))
    data_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                            generator=torch.Generator(device))
    optimiser = optimiser(model.parameters(), lr=1e-3)
    cur_epoch = 0
    if load_state is not None:
        cur_epoch = load_model_state(load_state, model, optimiser)
    start_time = time.time()
    _, axs = plt.subplots(3, 3, figsize=(15, 10))
    TRAIN_FEATURES, VALID_FEATUES = [], []
    for i in range(num_epochs):
        *train_metrics, train_epoch_metrics = list(go_for_epoch(
            data_train, batch_size, cur_epoch + i, f"Epoch {cur_epoch + i} train", model, loss_func, optimiser, segmentation))

        torch.cuda.empty_cache() # ГПУщка не хватает

        *test_metrics, test_epoch_metrics = list(go_for_epoch(
            data_test, batch_size, cur_epoch + i, f"Epoch {cur_epoch + i} valid", model, loss_func, segmentation=segmentation))
        TRAIN_FEATURES.append(train_epoch_metrics)
        VALID_FEATUES.append(test_epoch_metrics)
        if save_state and i % period_save_weights == 0:  # сохранение параметров модели
            save_model_state(model, optimiser, f"{model_title}_{cur_epoch + i}", cur_epoch + i)
        # графики обучения
        if not save_graph:
            continue
        plot_graphs_of_education(axs, model_title, train_metrics, test_metrics,
                                 logging_iters_train, logging_iters_valid)
        for x, label in enumerate(["loss", "accuracy", "fscore"]): 
            axs[2, x].clear()
            plot_data(axs[2, x], [range(cur_epoch, cur_epoch + i + 1)] * 2, [np.array(TRAIN_FEATURES)[:, x], np.array(VALID_FEATUES)[:, x]],
                    [f"Train {label}", f"Valid {label}"], title=f"{model_title} epoch {label}")
        plt.savefig(os.path.join("graphs", f"{model_title}.png"))
    
    save_model_state(model, optimiser, f"{model_title}_{cur_epoch + num_epochs}", cur_epoch + num_epochs)
    print(f"Training time: {round(time.time() - start_time)} seconds")

