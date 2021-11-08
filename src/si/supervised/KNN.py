from Modelo import Modelo
from si.util.util import l2_distance
import numpy as np

class KNN(Modelo):
    def __init__(self, n_neighboors, classification = True):
        super(KNN).__init__()#invocar o init do modelo
        self.n_neighboors = n_neighboors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighboors(self, x):
        distance = l2_distance(x,self.dataset.X)
        idxs_sort = np.argsort(distance)#os indices vao ser postos por ordem crescente
        return idxs_sort[:self.n_neighboors]#ate aos neighboors especificados

    def predict(self, x):
        assert self.is_fitted, 'Model must be fot before predict'
        neigbhoors = self.get_neighboors(x)#obtem os neighboors (pontos mais proximos)
        values = self.dataset.Y[neigbhoors].tolist()
        if self.classification:
            prediction = max(set(values), key=values.count)#retorna o que tem o valor maximo da label (a label que se repete mais vezes)
        else:
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        #ma: temos de usar porque pode formatar a previsao
        return accuracy_score(self.dataset.Y, y_pred)
