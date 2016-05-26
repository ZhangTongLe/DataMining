# -*- coding: utf-8 -*-
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
    L = data[:,cols-1]
    L = L.astype(int)
    return D,L

if __name__ == '__main__' or __name__ == 'test_tree':
    print __name__
    
     # 从文件中读取训练数据
    D, L = load()
    
     # 随机生成训练数据
    #D = np.random.randint(5, size=(100, 20)) 
    #D[:50] *= 2
    #L = np.repeat((0,1), 50)
    
     # 决策树训练器
    C = dmpy.supervised.tree.decision_tree_learner('C45')
    
     # 根据训练集生成训练树模型
    model = C.train_tree(D, L)
    
     # 显示输出训练树模型
    #list_name = ['age','income','student','credit_rating']
    #ci_name   = ['no','yes']
    list_name = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20']
    ci_name   = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
    model.print_tree(list_name,ci_name ,outfile = '/root/workspace/DMPY/src/test.png')
    
     # 应用数据集进行预测
    data           = np.array([1,1,1,2,1,2,0,1,0,2,1])
    class_lable    = model.apply(data)
    print class_lable
    
    
    
    
    