import numpy as np
from scipy import stats
from  copy import copy
import warnings
from si.util.scale import StandardScaler

class PCA:
    def __init__(self, n_components=2, using="svd"):
        self.n_components = n_components
        self.using = using

    def fit(self, dataset):#NAO E PRECISO
        pass

    def transform(self, dataset):

        X_scaled = StandardScaler().fit_transform(dataset)#Faz a normalização por standart scaler
        features_Scaled = X_scaled.X.T #features passam para as linhas
        if self.using == "svd":
            self.vectors, self.values, rv = np.linalg.svd(features_Scaled)
        else:
            cov_matrix = np.cov(features_Scaled)#estima a covariancia -> array com a covariancia das matrizes
            self.values, self.vectors = np.linalg.eig(cov_matrix)#valores proprios
        self.sorted_comp = np.argsort(self.values)[::-1] #Gera uma lista com os indexs das colunas ordenadas por importancia de componte
        self.s_value = self.values[self.sorted_comp] #Colunas dos valores e vetores sao reordenadas pelos indexs das colunas
        self.s_vector = self.vectors[:, self.sorted_comp]
        if self.n_components not in range(0, dataset.X.shape[1]+1):
            warnings.warn('Number of components is not between 0 and '+str(dataset.X.shape[1]))
            self.n_components = dataset.X.shape[1]
            warnings.warn('Number of components defined as ' + str(dataset.X.shape[1]))
        self.vetor_subset = self.s_vector[:, 0:self.n_components] #gera um conjunto apartir dos vetores e values ordenados
        X_reduced = np.dot(self.vetor_subset.transpose(), features_Scaled).transpose()
        return X_reduced

    def explained_variances(self):
        soma = np.sum(self.s_value)
        percent = []
        for i in self.s_value:
            percent.append(i / soma * 100)
        return np.array(percent)
        #self.values_subset = self.s_value[0:self.n_components]
        #return np.sum(self.values_subset), self.values_subset

    def fit_transform(self,dataset):
        x_reduced = self.transform(dataset)
        e_var = self.explained_variances()
        return x_reduced, e_var
