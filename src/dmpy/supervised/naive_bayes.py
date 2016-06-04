import numpy as np
import scipy as sp
from collections import Counter
import matplotlib.pyplot as plt
import itertools

__all__ = [
    'naive_bayes_learner',
    ]


class naive_bayes_learner(object):
    def __init__(self):
        self.D = None
        self.L = None
        pass
 
    def train_nb(self,D, L):
           self.D = D
           self.L = L
           
    def apply(self,D):
        pass