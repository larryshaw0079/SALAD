import time
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset

from .preprocessing import fill_missing_data, reconstruct_data, standardize_time_series


class KPIBatchedWindowDataset(Dataset):
    def __init__(self, series, label, mask, window_size=120, stride=1):
        super(KPIBatchedWindowDataset, self).__init__()
        self.series = series
        self.label = label
        self.mask = mask

        self.window_size = window_size
        self.stride = stride

        if len(self.series.shape) != 1:
            raise ValueError('The `series` must be an 1-D array!')

        if label is not None and (label.shape != series.shape):
            raise ValueError('The shape of `label` must agrees with the shape of `series`!')

        if mask is not None and (mask.shape != series.shape):
            raise ValueError('The shape of `mask` must agrees with the shape of `series`!')

        self.tails = np.arange(window_size, series.shape[0]+1, stride)

    def __getitem__(self, idx):
        x = self.series[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

        if (self.label is None) and (self.mask is None):
            # Only data

            return torch.from_numpy(x)
        elif self.mask is None:
            # Data and label
            y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

            return torch.from_numpy(x), torch.from_numpy(y)
        elif self.label is None:
            # Data and mask
            m = self.mask[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

            return torch.from_numpy(x), torch.from_numpy(m)
        else:
            y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
            m = self.mask[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

    def __len__(self):
        return self.tails.shape[0]


def prepare_dataset(data_path, data_category, train_val_test_split, label_portion, standardization_method='negpos1', filling_method='zero'):
    # Assertions
    assert (sum(train_val_test_split) == 10)
    assert (standardization_method in ['standard', 'minmax', 'negpos1', 'none'])
    assert (filling_method in ['prev', 'zero', 'none'])

    with open(data_path, 'r', encoding='utf8') as f:
        df = pd.read_csv(f)
    if data_category == 'kpi':
        time_stamp = df['timestamp'].values
        value = df['value'].values
        label = df['label'].values
    elif data_category == 'nab':
        time_stamp = np.array(list(map(lambda s: time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S')), df['timestamp'].values)))
        value = df['value'].values
        label = df['label'].values
    elif data_category == 'yahoo':
        if 'timestamp' in df.columns:
            time_stamp = df['timestamp'].values
        else:
            time_stamp = df['timestamps'].values
        value = df['value'].values
        if 'changepoint' in df.columns:
            label = np.logical_or(df['changepoint'].values, df['anomaly'].values)
        else:
            label = df['is_anomaly'].values
    else:
        raise ValueError('Invalid data category!')

    # Reconstruct data
    time_stamp, value, label, mask = reconstruct_data(time_stamp, value, label)
    # Filling missing data
    value = fill_missing_data(value, mask, method=filling_method)
    # Standardization
    value = standardize_time_series(value, standardization_method, mask=np.logical_or(label, mask))

    # datetimes = [datetime.fromtimestamp(time_stamp[i]) for i in range(len(time_stamp))]

    # Pre-processing
    quantile1 = train_val_test_split[0] / 10
    quantile2 = (10 - train_val_test_split[-1]) / 10

    train_x, train_y, train_m = value[:int(value.shape[0] * quantile1)], \
                                         label[:int(label.shape[0] * quantile1)], \
                                         mask[:int(mask.shape[0] * quantile1)]
    val_x, val_y, val_m = value[int(value.shape[0] * quantile1):int(value.shape[0] * quantile2)], \
                                 label[int(label.shape[0] * quantile1):int(label.shape[0] * quantile2)], \
                                 mask[int(mask.shape[0] * quantile1):int(mask.shape[0] * quantile2)]
    test_x, test_y, test_m = value[int(value.shape[0] * quantile2):], \
                                     label[int(label.shape[0] * quantile2):], \
                                     mask[int(mask.shape[0] * quantile2):]

    if quantile1 == quantile2:
        val_x = None
        val_y = None

    if label_portion == 0.0:
        train_y = np.zeros_like(train_y)
    else:
        anomaly_indices = np.arange(train_y.shape[0])[train_y == 1]
        selected_indices = np.random.choice(anomaly_indices,
                                            size=int(np.floor(anomaly_indices.shape[0] * (1 - label_portion))), replace=False)
        train_y[selected_indices] = 0

    return train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m
