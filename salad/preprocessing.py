import numpy as np
import pandas as pd


def reconstruct_data(time_stamp, values=None, label=None):
    """Reconstruct a single series according to the time stamp

    The missing parts will be imputed with zeroes

    :param time_stamp:  The time stamp
    :type time_stamp:   np.ndarray
    :param values:      The values, defaults to None
    :type values:       np.ndarray, optional
    :param label:       The labels indicating abnormality, defaults to None
    :type label:        np.ndarray, optional
    :raises ValueError: `time_stamp` must be an 1-D array!
    :raises ValueError: Duplicated values in `time_stamp`!
    :raises ValueError: Misunderstanding `time_stamp` intervals!
    :return:            Reconstructed time stamp series, reconstructed time series values (missing attributes are noted as None),
                        reconstructed labels (missing attributes are noted as None) and missing point indicators (missing: 1, normal: 0)
    :rtype:             tuple
    """
    time_stamp = np.asarray(time_stamp, np.int64)
    if len(time_stamp.shape) != 1:
        raise ValueError('`time_stamp` must be an 1-D array!')

    src_index = np.argsort(time_stamp)  # Sorted indices
    time_stamp_sorted = time_stamp[src_index]  # Sorted time stamps
    intervals = np.unique(np.diff(time_stamp_sorted))  # Time intervals
    min_interval = np.min(intervals)  # Minimum time interval
    if min_interval == 0:
        raise ValueError('Duplicated values in `time_stamp`!')

    # All the time intervals should be multipliers of the minimum time interval
    for interval in intervals:
        if interval % min_interval != 0:
            raise ValueError('Misunderstanding `time_stamp` intervals!')

    reconstructed_length = (time_stamp_sorted[-1] - time_stamp_sorted[0]) // min_interval + 1
    reconstructed_time_stamp = np.arange(time_stamp_sorted[0], time_stamp_sorted[-1] + min_interval, min_interval,
                                         dtype=np.int64)
    missing_indicators = np.ones([reconstructed_length], dtype=np.int32)
    dest_index = np.asarray((time_stamp_sorted - time_stamp_sorted[0]) // min_interval, dtype=np.int)
    missing_indicators[dest_index] = 0

    reconstructed_values = None
    if values is not None:
        values_sorted = values[src_index]
        reconstructed_values = np.zeros([reconstructed_length], dtype=values.dtype)
        reconstructed_values[dest_index] = values_sorted

    reconstructed_label = None
    if label is not None:
        label_sorted = label[src_index]
        reconstructed_label = np.zeros([reconstructed_length], dtype=label.dtype)
        reconstructed_label[dest_index] = label_sorted

    return reconstructed_time_stamp, reconstructed_values, reconstructed_label, missing_indicators


def fill_missing_data(data, mask, method='prev'):
    data[mask == 1] = np.nan
    if method == 'prev':
        series = pd.Series(data)
        series = series.interpolate(method='linear')

        return series.values
    elif method == 'zero':
        values = np.zeros_like(data)
        values[mask == 0] = data[mask == 0]

        return values
    elif method == 'none':
        return data
    else:
        raise ValueError('Invalid filling method!')


def standardize_time_series(values, method='minmax', mask=None):
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be an 1-D array!')

    if mask is not None and mask.shape != values.shape:
        raise ValueError('The shape of `mask` dose not agree with the shape of `values`!')

    if mask is not None:
        values_excluded = values[np.logical_not(mask)]
    else:
        values_excluded = values

    if method == 'none':
        pass
    elif method == 'minmax':
        min_value = np.min(values_excluded)
        max_value = np.max(values_excluded)

        return (values - min_value) / (max_value - min_value)
    elif method == 'standard':
        mean_value = np.mean(values_excluded)
        std_value = np.std(values_excluded)

        return (values - mean_value) / std_value
    elif method == 'negpos1':
        min_value = np.min(values_excluded)
        max_value = np.max(values_excluded)

        return ((values - min_value) / (max_value - min_value) - 0.5) / 0.5
    else:
        raise NotImplemented('Invalid standardize method!')


def get_windowed_data(series, window_size, stride):
    if len(series.shape) != 1:
        raise ValueError('The `series` must be an 1-D array!')

    for i in range(0, series.shape[0], stride):
        if i + window_size > series.shape[0]:
            break
        else:
            yield series[i: i + window_size]

