
from sklearn.base import BaseEstimator, BaseTransformer
import numpy as np

class LDAEnsemble(BaseEstimator, BaseTransformer):
    """
    LDAEnsemble does Fischer's linear discriminant aanlysis
    but calculates the between and within class variance in a 
    specific way. Each class is defined by a different ensemble 
    of structures. So the data consists of structures as well 
    as population weights.

    """
    def __init__(self):
        super(LDAEnsemble, self).__init__()
    
    
    def fit(self, X0, pop0, X1, pop1):
        """
        Fit the lda model. This will NOT work with sklearn's
        cross validation stuff. But we don't currently have
        any hyperparameters to fit anyway.

        Parameters
        ----------
        X0 : np.ndarray, shape = [n_conformations, n_features]
            Data for class 0.
        pop0 : np.ndarray, shape = [n_conformations]
            Weight of each conformation in ensemble 0.
        X1 : np.ndarray, shape = [n_conformations, n_features]
            Data for class 1.
        pop1 : np.ndarray, shape = [n_conformations]
            Weight of each conformation in ensemble 1.
        """

        # reshape to column vectors, and make sure
        # they're normalized to one
        pop0 = pop0.reshape((-1, 1)) / np.float(pop0.sum())
        pop1 = pop1.reshape((-1, 1)) / np.float(pop1.sum())

        self.mean0_ = pop0.T.dot(X0)
        self.mean1_ = pop1.T.dot(X1)

        self.sigma0_ = (X0 - mean0).T.dot((X0 - mean0) * pop0)
        self.sigma1_ = (X1 - mean1).T.dot((X1 - mean1) * pop1)

        self.coef_ = np.linalg.inv(self.sigma0 + self.sigma1).dot(self.mean0_ - self.mean1_)

