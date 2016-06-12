# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from collections import Counter
import matplotlib.pyplot as plt
import itertools

__all__ = [
    'naive_bayes_learner',
    ]

def _P_Ci_(L):
    '''
    计算每个类Ci的先验概率P(Ci) ,可以根据训练元组计算.
    '''
    #   class下每个种类ci的个数【dict】
    kind_of_cis_count = Counter(L.tolist())
    kind_of_ci_count_sum = sum([ci_count[1]for ci_count in kind_of_cis_count.items()])
    
    # class下有多少种类ci【list】
    kind_of_cis= kind_of_cis_count.keys()
    for kind_of_ci in kind_of_cis:               
        info_D = info_D  + (-1)*(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)*sp.log2(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)))
    return info_D
    pass

class naive_bayes_learner(object):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

    Parameters
    ----------
    alpha : float, optional (default=1.0) Additive (Laplace/Lidstone)平滑参数 (0 for no smoothing).

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    intercept_ : property
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    coef_ : property
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]

    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha

 
    def train_nb(self,D, L):
        self.D = D
        self.L = L
        
        _,col =D.shape
        assert len(L) == col
           
    def predict(self, X):
        _P_Ci_(self.L)
        
        

