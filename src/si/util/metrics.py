import numpy as np
import pandas as pd

def accuracy_score(y_true, y_pred):
    """Classification performance metric compute the accuracy of y_true and y_pred
    :param numoy.array y_true: like a shape array
    :param numoy.array y_pred: like a shape array
    :return c (float) accuracy score"""

    correct = 0
    for true, pred in zip(y_true,y_pred):
        if true == pred:
            correct += 1
    accuracy = correct/len(y_true)
    return accuracy

def mse(y_true, y_pred, squared = True):#Mean squared error
    """Mean squared errir regression loss fucntion.
    Parameters
    :param numpy.array y_true: array like of shape (n samples,)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred)**2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred)).sum()


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.
    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score

class ConfusionMatrix:

    def __call__(self, true_y, pred_y):
        self.true = np.array(true_y)
        self.pred = np.array(pred_y)
        return self.to_df()

    def calc(self):
        cm = pd.crosstab(self.true, self.pred, rownames=["Actual Values"], colnames=["Predicted Values"], margins=True)
        return cm

    def to_df(self):
        return pd.DataFrame(self.calc())
