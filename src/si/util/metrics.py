import numpy as np


def mse(y_true, y_pred, squared = True):
    """Mean squared errir regression loss fucntion.
    Parameters
    :param numpy.array y_true: array like of shape (n samples,)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred)**2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)
