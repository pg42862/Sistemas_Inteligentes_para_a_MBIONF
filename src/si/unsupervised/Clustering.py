import numpy as np
from scipy import stats
from  copy import copy
import warnings
from si.util.util import l2_distance
from src.si.util.util import euclidean,manhattan

class Kmeans:
    def __init__(self, K: int, max_interactions = 1000, distance = 'euclidean'):
        self.k = K
        self.max_interaction = max_interactions
        self.centroids = None
        if distance == 'euclidean':
            self.distance = euclidean
        elif distance == 'manhattan':
            self.distance = manhattan
        else:
            raise Exception('Distance metric not available \n Score functions: euclidean, manhattan')
    
    def fit(self, dataset):
        """Randomly selects K centroids"""

        x = dataset.X
        self._min = np.min(x, axis = 0)
        self._max = np.max(x, axis = 0)

    def init_centroids(self, dataset):
        x = dataset.X
        self.centroids = np.array(
            [np.random.uniform(low=self._min[1], high=self._max[1], size=(self.k,)
        ) for i in range(x.shape[1])]).T

    def get_closest_centroid(self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index
    
    def transform(self, dataset):
        self.init_centroids(dataset)
        X = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])
        #data.shape() -> The elements of the shape tuple give the lengths of the corresponding array dimensions.
        while changed or count < self.max_interaction:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=X.T)
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis=0))#calcular a media sobre os pontos e essas medias vao ser os novos pontos
            #cent = [np.mean(X[idxs == i],axis = 0) for i in range(self.k)]
            self.centroids = np.array(cent)
            changed = np.all(old_idxs == idxs) #Test whether all array elements along a given axis evaluate to True.
            old_idxs = idxs
            count += 1
        return self.centroids, idxs

    def fit_trasform(self, dataset):
        self.fit(dataset)
        centroides, idxs = self.transform(dataset)
        return centroides, idxs