from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from data.dataset import StandardDataset


def compile_metrics(dataset: StandardDataset, predictions: np.ndarray, timestamps: List,
                    thresholds: List[float]) -> Tuple[np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param dataset:
    :param predictions: shape=(batch_size, height, width), dtype=int
    :param timestamps:
    :param thresholds:
    :return: confusion_matrix, binary_metrics_by_threshold

    confusion: (prediction, target)

    pred\target |   false  |   true
    false       |   cn     |   miss
    true        |   fa     |   hit
    """
    assert (predictions.shape[0] == len(timestamps))
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(predictions, torch.Tensor):
        raise ValueError("`predictions` must be a numpy array")

    n_classes = len(thresholds) + 1
    padded_thresholds = [float('-inf')] + thresholds + [float('inf')]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int)
    binary_metrics_by_threshold = defaultdict(lambda: defaultdict(list))
    for (origin, lead_time), prediction in zip(timestamps, predictions):
        local_confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int)
        target = dataset.aws_base_dataset.load_array(origin + timedelta(hours=lead_time), 0)
        valid = target != -9999

        # Populate local confusion matrix
        interval_list = zip(padded_thresholds[:-1], padded_thresholds[1:])
        preds_by_interval = []
        targets_by_interval = []
        for i, (start, end) in enumerate(interval_list):
            preds_by_interval.append(prediction == i)
            targets_by_interval.append((start <= target) & (target < end))
        for i, p in enumerate(preds_by_interval):
            for j, t in enumerate(targets_by_interval):
                local_confusion_matrix[i, j] = (valid & p & t).sum()

        # Populate binary metric dict[list]s
        for i, threshold in enumerate(thresholds, start=1):
            metrics = binary_metrics_by_threshold[threshold]
            metrics["origin"].append(origin.strftime("%Y%m%d_%H%M"))
            metrics["lead_time"].append(lead_time)
            metrics["hit"].append(local_confusion_matrix[i:, i:].sum())
            metrics["miss"].append(local_confusion_matrix[:i, i:].sum())
            metrics["fa"].append(local_confusion_matrix[i:, :i].sum())
            metrics["cn"].append(local_confusion_matrix[:i, :i].sum())

        confusion_matrix += local_confusion_matrix

    binary_metrics_by_threshold = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold.items()}
    return confusion_matrix, binary_metrics_by_threshold


def get_binary_confusion_matrix_dict(confusion_matrix: np.ndarray, threshold_index: int):
    """
    Deprecated (as of 220509)
    Multi-class confusion matrix를 주어진 threshold_index에 대해 2x2 binary matrix로 변환합니다.

    :param confusion_matrix:
    :param threshold_index:
    :return:
    """
    assert type(confusion_matrix) != pd.DataFrame, 'you must pass a numpy array'
    i = threshold_index
    binary_cm = np.zeros([2, 2])  # actual x pred (0=False, 1=True)

    binary_cm[1, 1] = confusion_matrix[i:, i:].sum()  # true positive (hit)
    binary_cm[0, 0] = confusion_matrix[:i, :i].sum()  # true negative (correct negative)
    binary_cm[1, 0] = confusion_matrix[i:, :i].sum()  # false negative (miss)
    binary_cm[0, 1] = confusion_matrix[:i, i:].sum()  # false positive (false alarm)

    return {
        'hit': binary_cm[1, 1].sum(),  # true positive (hit)
        'miss': binary_cm[1, 0].sum(),  # false negative (miss)
        'fa': binary_cm[0, 1].sum(),  # false positive (false alarm)
        'cn': binary_cm[0, 0].sum(),  # true negative (correct negative)
    }


def compute_measures_for_confusion_dict(confusion_dict, in_place=True):
    """
    Deprecated (as of 220504) - you should use DataFrames when working with measures.
    Use `compute_measures_for_dataframe` instead.

    get_confusion_dict에서 출력한 dict를 바탕으로 각종 성능 measure를 계산합니다.

    :param confusion_dict:
    :param in_place: True 시, 기존 dict에 결과를 추가합니다
    :return:
    """
    d = confusion_dict
    hit, miss, fa, cn = d['hit'], d['miss'], d['fa'], d['cn']

    measures = {
        'acc': (hit + cn) / (hit + miss + fa + cn),  # accuracy
        'pod': hit / (hit + miss),  # probability of detection
        'csi': hit / (hit + miss + fa),  # critical success index
        'far': fa / (hit + fa),  # false alarm ratio
        'bias': (hit + fa) / (hit + miss),  # bias
    }

    if in_place:
        d.update(measures)
        return d
    else:
        return measures


def compute_evaluation_metrics(df: pd.DataFrame):
    df['acc'] = (df['hit'] + df['cn']) / (df['hit'] + df['miss'] + df['fa'] + df['cn'])  # accuracy
    df['pod'] = df['hit'] / (df['hit'] + df['miss'])  # probability of detection
    df['far'] = df['fa'] / (df['hit'] + df['fa'])  # false alarm ratio
    df['csi'] = df['hit'] / (df['hit'] + df['miss'] + df['fa'])  # critical success index
    df['bias'] = (df['hit'] + df['fa']) / (df['hit'] + df['miss'])  # bias
    return df
