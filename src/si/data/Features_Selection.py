import numpy as np
from scipy import stats
from  copy import copy
import warnings

class VarianceThreshold:
    def __init__(self, threshold=0):
        """The variance threshold os a simple baseline approach to feat"""
        if threshold <0:
            warnings.warn("The thereshold must be a non-negative value.")
        self.threshold = threshold
    
    def fit(self, dataset):
        """Calcula a variancia"""
        X = dataset.X
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        """Escolhe as variancias que sao maiores que o threshold"""
        X = dataset.X
        cond = self._var > self.threshold #array de booleanos se esta cond se verifica
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:,idxs] #:->todas as linhas, idxs -> features que me interessa manter
        xnames = [dataset._xnames[i] for i in idxs]
        if inline:#se for inline
            dataset.X = X_trans #ubstituir as variaveis
            dataset._xnames = xnames #atualizo os nomes
            return dataset
        else:
            from .dataset import Dataset
            return Dataset
