import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def reconstruct_predict(y_pred, y_true, delay=7, preserve_score=False):
    """Compute modified labels for a single series

    :param y_pred: The predictions, can only be an 1-d array
    :type y_pred:  np.ndarray
    :param y_true: The ground truth, can only be an 1-d array
    :type y_true:  np.ndarray
    :param delay:  The tolerance interval, defaults to 7
    :type delay:   int, optional
    :return:       Modified labels
    :rtype:        np.ndarray
    """
    change_points = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = (y_true[0] == 1)
    modified_y_pred = np.array(y_pred)
    current_pos = 0

    for point in change_points:
        if is_anomaly:
            if 1 in y_pred[current_pos: min(current_pos + delay + 1, point)]:
                if preserve_score:
                    modified_y_pred[current_pos:point] = np.max(modified_y_pred[current_pos:point])
                else:
                    modified_y_pred[current_pos:point] = 1
            else:
                modified_y_pred[current_pos:point] = 0

        is_anomaly = not is_anomaly
        current_pos = point

    point = len(y_true)

    if is_anomaly:
        if 1 in y_pred[current_pos: min(current_pos + delay + 1, point)]:
            modified_y_pred[current_pos:point] = 1
        else:
            modified_y_pred[current_pos:point] = 0

    return modified_y_pred


def get_modified_predict_label(y_preds, y_trues, delay=7, preserve_score=False):
    """Compute modified labels

    :param y_preds:     The predictions, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_preds:      np.ndarray
    :param y_trues:     The ground truth, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_trues:      np.ndarray
    :param delay:       The tolerance interval, defaults to 7
    :type delay:        int, optional
    :raises ValueError: Invalid inputs!
    :return:            Modified labels
    :rtype:             np.ndarray
    """
    assert (type(y_preds) == type(y_trues))
    if isinstance(y_preds, np.ndarray) and len(y_preds.shape) == 1:
        # 1D array
        y_pred = reconstruct_predict(y_preds.reshape(-1), y_trues.reshape(-1), delay, preserve_score)
        y_true = y_trues.reshape(-1)
    elif isinstance(y_preds, np.ndarray) and len(y_preds.shape) == 2:
        # 2D array
        raise NotImplementedError
        # y_pred = []
        # for i in range(y_preds.shape[0]):
        #     y_pred.append(reconstruct_predict(y_preds[i].reshape(-1), y_trues[i].reshape(-1), delay, preserve_score))
        # y_pred = np.concatenate(y_pred)
        # y_true = np.concatenate(y_true, axis=0)
    elif isinstance(y_preds, list):
        # List
        y_pred = []
        for i, data in enumerate(y_preds):
            y_pred.append(reconstruct_predict(y_preds[i].reshape(-1), y_trues[i].reshape(-1), delay, preserve_score))
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_trues)
    else:
        raise ValueError('Invalid inputs!')

    return y_pred, y_true


def modified_precision(y_pred, y_true, mask=None, delay=7):
    """Compute the segment-oriented precision score

    :param y_pred:      The predictions, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_pred:       list or np.ndarray
    :param y_true:      The ground truth, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_true:       list or np.ndarray
    :param mask:        The mask indicating missing elements (missing: 1, normal: 0), defaults to None
    :type mask:         list or np.ndarray, optional
    :param delay:       The tolerance interval, defaults to 7
    :type delay:        int, optional
    :raises ValueError: Invalid mask type!
    :return:            The modified precision score
    :rtype:             np.ndarray
    """
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            # 1-D or 2-D numpy array
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        elif isinstance(mask, list):
            mask = np.concatenate(mask)
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        else:
            raise ValueError('Invalid mask type!')

    return precision_score(y_true, y_pred)


def modified_recall(y_pred, y_true, mask=None, delay=7):
    """Compute the segment-oriented recall score

    :param y_pred:      The predictions, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_pred:       list or np.ndarray
    :param y_true:      The ground truth, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_true:       list or np.ndarray
    :param mask:        The mask indicating missing elements (missing: 1, normal: 0), defaults to None
    :type mask:         list or np.ndarray, optional
    :param delay:       The tolerance interval, defaults to 7
    :type delay:        int, optional
    :raises ValueError: Invalid mask type!
    :return:            The modified recall score
    :rtype:             np.ndarray
    """
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            # 1-D or 2-D numpy array
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        elif isinstance(mask, list):
            mask = np.concatenate(mask)
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        else:
            raise ValueError('Invalid mask type!')

    return recall_score(y_true, y_pred)


def modified_f1(y_pred, y_true, mask=None, delay=7):
    """Compute the segment-oriented f1 score

    :param y_pred:      The predictions, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_pred:       list or np.ndarray
    :param y_true:      The ground truth, can be an 1-d array, 2-d array or a list of 1-d arrays
    :type y_true:       list or np.ndarray
    :param mask:        The mask indicating missing elements (missing: 1, normal: 0), defaults to None
    :type mask:         list or np.ndarray, optional
    :param delay:       The tolerance interval, defaults to 7
    :type delay:        int, optional
    :raises ValueError: Invalid mask type!
    :return:            The modified f1 score
    :rtype:             np.ndarray
    """
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            # 1-D or 2-D numpy array
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        elif isinstance(mask, list):
            mask = np.concatenate(mask)
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        else:
            raise ValueError('Invalid mask type!')

    return f1_score(y_true, y_pred)


def modified_auc_score(y_pred, y_true, mask=None, delay=7):
    y_pred, y_true = get_modified_predict_label(y_pred, y_true, delay, True)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            # 1-D or 2-D numpy array
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        elif isinstance(mask, list):
            mask = np.concatenate(mask)
            y_pred = y_pred[mask == 0]
            y_true = y_true[mask == 0]
        else:
            raise ValueError('Invalid mask type!')

    return roc_auc_score(y_true, y_pred)


def range_lift_with_delay(array: np.ndarray, label: np.ndarray, delay=None, inplace=False) -> np.ndarray:
    """
    :param delay: maximum acceptable delay
    :param array:
    :param label:
    :param inplace:
    :return: new_array
    """
    assert np.shape(array) == np.shape(label)
    if delay is None:
        delay = len(array)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_array = np.copy(array) if not inplace else array
    pos = 0
    for sp in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, sp)
            new_array[pos: ptr] = np.max(new_array[pos: ptr])
            new_array[ptr: sp] = np.maximum(new_array[ptr: sp], new_array[pos])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        ptr = min(pos + delay + 1, sp)
        new_array[pos: sp] = np.max(new_array[pos: ptr])
    return new_array
