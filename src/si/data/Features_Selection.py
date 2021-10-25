import numpy as np
from scipy import stats
from  copy import copy
import warnings

from si.data.dataset import Dataset
__all__ = ['Dataset']

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
            dataset.X = X_trans #substituir as variaveis
            dataset._xnames = xnames #atualizo os nomes
            return dataset
        else:
            from .dataset import Dataset
            return Dataset
    
    def fit_transform(self,dataset, inline=False):
        """Reduce X to the selected features."""
        self.fit(dataset)
        return self.tansform(dataset, inline=inline)


#F-Score(ANOVA): se as distribuições têm ou não a mesma média -> obtem a estatística F e os p-values
#Implementar 2 maneiras:
#I)
def f_classif(dataset):
    """Scoring fucntion for classification. Compute the ANOVA F-value for the provided sample.

			param dataset: A labeled dataset
			type dataset: Dataset
			return: F scores and p-value
                statistic F: The computed F statistic of the test.
                p_value: The associated p-value from the F distribution.
			rtype_ a tupple of np.arrays"""

    X, y = dataset.X, dataset.y
    args = []
    for k in np.unique(y):
        args.append(X[y == k, :])
    F_stat, p_value = stats.f_oneway(*args)#Perform one-way ANOVA.
        #The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. 
        #The test is applied to samples from two or more groups, possibly with differing sizes.
        #*args = sample1, sample2, …array_like:
            #The sample measurements for each group. There must be at least two arguments. 
            #If the arrays are multidimensional, then all the dimensions of the array must be the same except for axis.
    return F_stat, p_value

#II)
def f_regression(dataset):
    """Scoring function for regressions.

		param dataset: A labeled dataset
		type dataset: Dataset
		return: F scores and p-value"""

    X, y = dataset.X, dataset.y
    correlation_coef = np.array([stats.pearsonr(X[:,1], y)[0]])#X and y are array's
    #Pearson correlation coefficient and p-value for testing non-correlation:

        #The Pearson correlation coefficient measures the linear relationship between two datasets. 
        #The calculation of the p-value relies on the assumption that each dataset is normally distributed. 
        #Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. 
        #Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. 
        #Negative correlations imply that as x increases, y decreases.
        #The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a 
        #Pearson correlation at least as extreme as the one computed from these datasets.
        
            #return: 
                    #r:Pearson’s correlation coefficient.
                    #p-value:Two-tailed p-value.

    degree_of_freedom = y.size - 2
    corr_coef_squared = correlation_coef **2
    F_stat = corr_coef_squared / (1 - corr_coef_squared) * degree_of_freedom
    p_value = stats.f.sf(F_stat, 1, degree_of_freedom) 
    #sf(x, dfn, dfd, loc=0, scale=1) -> Survival function (or reliability function or complementary cumulative distribution function): 
                                        #The survival function is a function that gives the probability that a patient, 
                                        #device, or other object of interest will survive beyond any specified time.
    #dnf -> Disjunctive normal form
    #dfd -> Degrees of freedom
    return F_stat, p_value


class SelectKBest:
    """"Select features according to the K(Number of top features) highest scores (removes all but the K highest scoring features)."""
    def __init__(self, score_function, K):
        """
        Parameters:
            score_func: Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. 
            K: int or “all” -> Number of top features to select."""

        available_score_function = ["f_classif", "f_regression"]
        
        #Ciclo para escolher funçao de score
        if score_function not in available_score_function:
            raise Exception(f"Scoring function not available. Please choose between: {available_score_function}.")
        elif score_function == "f_classif":
            self.score_function = f_classif
        else:
            self.score_function = f_regression

        if K <= 0:
            raise Exception("The K value must be higher than 0.")
        else:
            self.k = K

        def fit(self, dataset):
            """Run score function on dataset and get the appropriate features."""
            self.F_stat, self.p_value = self.score_function(dataset) #score_function = f_classif or f_regression

        def transform(self, dataset, inline=False):
            """Reduce X to the selected features."""
            X, X_names = dataset.X, dataset.xnames

            if self.k > X.shape[1]:#se o K(numero de top features) for maior que o numero de features em X
                warnings.warn("The K value provided is greater than the number of features. "
                              "All features will be selected")
                self.k = int(X.shape[1])#selecionar todas as features

            select_features = np.argsort(self.F_stat)[-self.k:]

            X_features = X[:, select_features] #:->todas as linhas, select_features -> features slecionadas
            #X_features_names = [X_names[index] for index in select_features]
            X_features_names = []
            for index in select_features:
                X_features_names.append(X_names[index])

            if inline:
                dataset.X = X_features
                dataset.xnames = X_features_names
                return dataset
            else:
                return Dataset(X_features, copy(dataset.Y), X_features_names, copy(dataset.yname))

        def fit_transform(self, dataset, inline=False):
            """Fit to data, then transform it.
            Fits transformer to X and y and returns a transformed version of X."""
            self.fit(dataset)
            return self.transform(dataset, inline=inline)