from Model import Model
from si.util.util import euclidean
from src.si.util.util import accuracy_score
import numpy as np

class KNN(Model):
    def __init__(self, n_neighbors=5, classification=True):
        super(KNN).__init__()#invocar o init do modelo
        self.k_neighbors = n_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        dist = euclidean(x, self.dataset.X)
        idx_sort = np.argsort(dist)#os indices vao ser postos por ordem crescente
        return idx_sort[:self.k_neighbors]#ate aos neighboors especificados

    def predict(self, x):
        assert self.is_fitted, 'Model must be fitted before prediction'
        neighbors = self.get_neighbors(x)#obtem os neighboors (pontos mais proximos)
        values = self.dataset.Y[neighbors].tolist()
        if self.classification:
            prediction = max(set(values), key=values.count)#retorna o que tem o valor maximo da label (a label que se repete mais vezes)
        else:
            prediction = sum(values) / len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        # ma: temos de usar porque pode formatar a previsao
        return accuracy_score(self.dataset.Y, y_pred)
