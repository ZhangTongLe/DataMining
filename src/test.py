'''
Created on 2016-5-4

@author: root
'''
import dmpy.supervised.tree
import os
import numpy as np
import matplotlib.pyplot as plt


def load(force_contiguous=True):
    _winedatafile = os.path.dirname(__file__)+'/wine.data'
    data  = np.array([list(map(float,line.split(','))) for line in open(_winedatafile)])
    L = data[:,0]
    L = L.astype(int)
    D = data[:,1:]
    D = D.astype(int)
    return D,L

if __name__ == '__main__' or __name__ == 'test_tree':
    print __name__
    
    C = dmpy.supervised.tree.decision_tree_learner('CART')
    D, L = load()
    root = C.train_tree(D, L)
    print C.tree_w_d(root)
    list_name = ['age','income','student','credit_rating']
    ci_name   = ['no','yes']
    C.print_tree(root,list_name,ci_name)
    pass