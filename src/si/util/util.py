import itertools
import pandas as pd
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n):#da nome as colunas
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

    return [gen() for _ in range(n)]#retorna uma lista com os nomes

def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    param dataset: A Dataset object
    type dataset: si.data.Dataset
    param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    type format: str, optional
    """
    if dataset.hasLabel():#verifica se existe label
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y),1)))
        #np.hstack(junta X,Y) -> reshape em self.Y.reshape(len(self.Y): linhas e 1))) coluna
        names = []#lista com nome das colunas
        for i in dataset._xnames:
            names.append(i)
        names.append(dataset._yname)#adicionar o nome da coluna label
    else:#se nao tiver label
        data = dataset.X.copy()
        names = [dataset._xnames]# -> names = [[dataset._xnames]]: nomes das colunas das variaveis indepedentes
    mean = np.mean(data, axis=0)#axis 0 = rows, axis 1 = columns
    var = np.var(data, axis=0)
    maxim = np.max(data, axis=0)
    minim = np.min(data, axis=0)
    stats = {}
    #-> stats ={names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}, names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}}
    for i in range(data.shape[1]):#percorre as colunas
        stat = {'mean': mean[i]#faz a media da coluna i
            ,'var': var[i]#faz a variancia da coluna i
            ,'max': maxim[i]#faz o maximo da coluna i
            ,'min': minim[i]}#faz o minimo da coluna i

        stats[names[i]] = stat #key: names[i], value: stat
    if format == 'df':#se quiser em pandas dataframe
        df = pd.DataFrame(stats)#convert an array to a dataframe
        return df
    else:#se nao quiser
        return stats #retorna o dicionario stats

def euclidean(x,y): #vai calcular a distancia de euclidean
    distance = np.sqrt(np.sum((x - y)**2, axis=1))
    return distance

def manhattan(x,y): #vai calcular a distancia de manhattan
    distance = (np.absolute(x-y)).sum(axis=1)
    return distance

def l2_distance(x,y): #vai calcular a distancia de euclidean
    distance = ((x - y)**2).sum(axis=1)
    return distance

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

def train_test_split(dataset, split = 0.8):
    n = dataset.X.shape[0]#numero de linhas
    m = int(split*n)#faz a conta
    arr = np.arange(n)#Return evenly spaced values within a given interval.
    np.random.shuffle(arr)
    from ..data import Dataset
    train = Dataset(dataset.X[arr[:m]], dataset.Y[arr[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[arr[m:]], dataset.Y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test

def sigmoid(x):
    return 1/(1+np.exp(-x))

def add_intersect(X):
    return np.hstack((np.ones((X.shape[0],1))),X)
