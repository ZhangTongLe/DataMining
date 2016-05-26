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
    rows,cols = data.shape
    D = data[0:,:cols-1]
    D = D.astype(int)
    XX,XXX = D.shape
    L = data[:,cols-1]
    L = L.astype(int)
    return D,L

if __name__ == '__main__' or __name__ == 'test_tree':
    print __name__
    D, L = load() 
    C = dmpy.supervised.tree.decision_tree_learner('C45')
    root = C.train_tree(D, L)
    print C.tree_w_d(root)
    #list_name = ['age','income','student','credit_rating']
    #ci_name   = ['no','yes']
    list_name = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
    ci_name   = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
    C.print_tree(root,list_name,ci_name)
    pass