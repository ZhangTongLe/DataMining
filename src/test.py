# -*- coding: utf-8 -*-
'''
Created on 2016-5-4

@author: root
'''
import dmpy.supervised.tree
import os
import numpy as np
import matplotlib.pyplot as plt
import dmpy.supervised.naive_bayes as nb


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
    
    
    '''
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X, y)
    A = [3, 3, 3, 7, 4, 3, 0, 3, 1, 2, 0, 2, 4, 2, 0, 3, 1, 3, 4, 2, 3, 0,1, 8, 4, 2, 1, 0, 1, 4, 1, 2, 5, 3, 2, 2, 4, 3, 1, 2, 4, 2, 1, 0,2, 3, 0, 4, 2, 4, 43, 4, 4, 3, 2, 4, 3, 33,34, 2, 1,32, 2, 3, 3, 3,4, 31, 0, 3, 4, 44, 1, 2, 1, 40, 2, 3, 4, 4, 2, 0, 1, 4, 24, 3, 1, 4,2, 4, 34, 2, 2, 1, 3, 2, 4, 0, 50, 2]
    print(clf.predict(A))
    '''
    
    
     # 从文件中读取训练数据
    D, L = load()
    
     # 朴素贝叶斯 
    NB = dmpy.supervised.naive_bayes.naive_bayes_learner()
    NB.train_nb(D,L)
    
     # 随机生成训练数据
    #D = np.random.randint(5, size=(100, 20)) 
    #D[:50] *= 2
    #L = np.repeat((0,1), 50)
    
     # 决策树训练器ID3 , C45 , CART 可选
    criterion = 'C45'
    C = dmpy.supervised.tree.decision_tree_learner(criterion)
    
     # 根据训练集生成训练树模型
    model = C.train_tree(D, L)
    
     # 显示输出训练树模型
    #list_name = ['age','income','student','credit_rating']
    #ci_name   = ['no','yes']
    list_name = ['A0','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19']
    ci_name   = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12']
    model.print_tree(list_name,ci_name ,outfile = '/root/workspace/DMPY/src/'+criterion+'.png')
    
     # 应用数据集进行预测
    data           = np.array([1,1,1,2,1,2,0,1,0,2,1])
    class_lable    = model.apply(data)
    print class_lable
    
     # 训练误差（再代入误差）
    apply_error , e_T_= model.apply_error(D, L)
    print "apply_error_rate=",apply_error,"e(T)=",e_T_
    
     # 悲观误差评估 
    pe = model.pessimistic_error(D, L,1.0/20.0)
    print "pessimistic_error=",pe
    
    # MDL误差评估
    mdl = model.MDL(D,L)
    print "MDL =" ,mdl  
     
     # 悲观剪枝
    model.PEP(D, L,1./20.0)
    model.print_tree(list_name,ci_name ,outfile = '/root/workspace/DMPY/src/'+criterion+'_PEP.png')
    
      # 悲观误差评估 
    pe = model.pessimistic_error(D, L,1.0/20.0)
    print "pessimistic_error=",pe   
    
    