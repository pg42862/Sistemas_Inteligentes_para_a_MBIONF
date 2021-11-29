from ..util.util import train_test_split
import numpy as np
import itertools

class CrossValidationScore:
    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv',3)
        self.split = kwargs.get('split',0.8)
        self.train_scores= None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_score = []
        test_score = []
        ds = []
        for _ in range(self.cv): #O _ serve para representação do valor não variável (não guarda)
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_score.append(self.model.cost())
                test_score.append(self.model.cost(test.X, test.Y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis = 0, arr = train.X.T)
                train_score.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis = 0, arr = train.X.T)
                test_score.append(self.score(train.Y, y_train))
        self.train_score = train_score #Guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.test_score = test_score #Guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.ds = ds
        return train_score, test_score

    def toDataFrame(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run function"
        return pd.DataFrame({"Train Scores:" : self.train_scores, "Test Scores:" : self.test_scores})

class GridSearchCV:
    def __init__(self, model, dataset, parameters, **kwargs):
        """
        :param model:
        :param dataset:
        :param parameters: dictionary with parameters
        :param kwargs:
        """
        self.model = model
        self.dataset = dataset
        hasparam = []
        hasparam = [hasattr(self.model, param) for param in parameters] #lista com os parametros
        #The hasattr() method returns true if an object has the given named attribute and false if it does not.

        if np.all(hasparam):#se todos forem atributos - True
            self.parameters = parameters
        else: # se houver algum que nao seja atributo (nao existir) vai dar um erro
            index = hasparam.index(False)
            #The index() method returns the index of the specified element in the list.
            keys = list(parameters.keys())
            #Dictionary1 = {'A': 'Geeks', 'B': 'For', 'C': 'Geeks'} -> dict_keys(['A', 'B', 'C'])
            #returns a view object that displays a list of all the keys in the dictionary in order of insertion.
            raise ValueError(f'Warning parameters: {keys[index]}')
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*values):
            for i in range(len(attrs)):
                setattr(self.model, attrs[i],conf[i])
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf,scores))
        return self.results

    def toDataframe(self):
        import pandas as pd
        assert self.results, "The grid search needs to be ran."
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = []
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            treino, teste = [], []
            for r in self.results:
                treino.append(r[1][0][i])
                teste.append(r[1][1][i])
            data['Train ' + str(i + 1)] = treino
            data['Test ' + str(i + 1)] = teste
        return pd.DataFrame(data)
