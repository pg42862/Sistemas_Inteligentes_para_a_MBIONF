import itertools
import pandas as pd
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]

def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    param dataset: A Dataset object
    type dataset: si.data.Dataset
    param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    type format: str, optional
    """
    if dataset.hasLabel():
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y),1)))
        names = []
        for i in dataset._xnames:
            names.append(i)
        names.append(dataset._yname)
    else:
        data = dataset.X.copy()
        names = [dataset._xnames]# -> names = [[dataset._xnames]]
    mean = np.mean(data, axis=0)#axis 0 = rows (“first” axis), axis 1 = columns
    var = np.var(data, axis=0)
    maxim = np.max(data, axis=0)
    minim = np.min(data, axis=0)
    stats = {}
    #-> stats ={names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}, names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}}
    for i in range(data.shape[1]):
        #data.shape() -> The elements of the shape tuple give the lengths of the corresponding array dimensions.
        stat = {'mean': mean[i]#faz a media daquela coluna
            , 'var': var[i]#faz a variancia da coluna i
            , 'max': maxim[i]#faz o maximo da coluna i
            , 'min': minim[i]}#faz o minimo da coluna i
        stats[names[i]] = stat #key: names[i], value: stat
    if format == 'df':#se quiser em pandas dataframe
        df = pd.DataFrame(stats)
        return df
    else:
        return stats

def euclidean(x,y): #vai calcular a distancia de euclidean
    distance = np.sqrt(np.sum((x - y)**2, axis=1))
    return distance

def manhattan(x,y): #vai calcular a distancia de manhattan
    distance = (np.absolute(x-y)).sum(axis=1)
    return distance
