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
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime,timedelta
from matplotlib.dates import AutoDateLocator, DateFormatter  
import matplotlib.dates as dt  

def load(force_contiguous=True):
    _winedatafile = os.path.dirname(__file__)+'/wine.data'
    data  = np.array([list(map(float,line.split(','))) for line in open(_winedatafile)])
    rows,cols = data.shape
    D = data[0:,:cols-1]
    D = D.astype(int)
    L = data[:,cols-1]
    L = L.astype(int)
    return D,L



def print_Livermore(datafile,start_date,end_date,key_value) :
    
    _winedatafile = datafile
    data1  = np.array([list(map(float,line.split(','))) for line in open(_winedatafile)])
    rows,cols = data1.shape
    D = data1[:,0]
    D = D.astype(int)
    D = D.tolist()
    
    
    L = data1[:,4]
    L = L.astype(int)   
    L = L.tolist()
    D_L = np.array([D,L])
    D_L = D_L.transpose()
    
    dates=[]
    s_sub = -1
    e_sub = -1
    sub   = 0
    for i in D_L:
        if end_date == 0 :
            if i[0] == start_date :
                s_sub = sub
                e_sub = 0
                break
        else : 
            if i[0] == start_date :
                s_sub = sub
            if i[0] == end_date :
                e_sub = sub
            if e_sub!=-1 and s_sub!=-1:
                break
        sub = sub+1                 
    D_L = D_L[e_sub:s_sub]
 
    D = D_L[:,0]
    for i in D:
        str = repr(i)
        year  = int(str[0])*1000+int(str[1])*100+int(str[2])*10+ int(str[3])
        month = int(str[4])*10+int(str[5])
        day   = int(str[6])*10+int(str[7])
        dates = dates+[datetime(year, month, day, 0, 0, 0,0)]
    D = dates
    L = D_L[:,1]

    
     # 画布
    fig = plt.figure(figsize=(39,13))       
      # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax = fig.add_subplot(111)
      # 坐标
    ax.grid()
    
    D_max = max(D)
    D_min = min(D)
    L_max = max(L)
    L_min = min(L)
    
    
    autodates = AutoDateLocator()  
    yearsFmt = DateFormatter('%Y-%m-%d')  
    fig.autofmt_xdate()        #设置x轴时间外观  
    ax.xaxis.set_major_locator(autodates)       #设置时间间隔  
    ax.xaxis.set_major_formatter(yearsFmt)      #设置时间显示格式  



    l=[D_min,D_max,L_min,L_max]  
    plt.axis(l)
    
    
    ax.plot(D,L.tolist())
      
     # 保存成图片
    plt.show()  
    plt.savefig('/root/workspace/DMPY/src/dd.jpg', dpi=240)

if __name__ == '__main__' or __name__ == 'test_tree':    
    print __name__
    print_Livermore(os.path.dirname(__file__)+'/BLC.csv',20160304,0,6) 
    #X = np.random.randint(5, size=(6, 100))
    X = np.array([[0, 2, 3, 3, 0, 3, 2, 3, 4, 3, 0, 3, 2, 1, 3, 3, 4, 0, 2, 3, 0, 2,2, 3, 4, 4, 2, 1, 4, 0, 0, 2, 3, 4, 3, 3, 4, 4, 2, 3, 3, 0, 3, 2,2, 2, 4, 4, 3, 0, 1, 3, 4, 0, 0, 4, 1, 3, 3, 2, 3, 0, 2, 4, 3, 0,3, 0, 3, 1, 4, 1, 3, 2, 2, 1, 1, 0, 0, 2, 3, 1, 1, 4, 1, 3, 2, 3,2, 1, 1, 3, 0, 4, 0, 3, 1, 0, 0, 0],[2, 2, 2, 2, 0, 4, 3, 2, 3, 3, 1, 2, 2, 1, 1, 4, 0, 1, 3, 4, 0, 1,2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 2, 4, 1, 3, 2, 2, 0, 3, 1, 1, 0, 1,2, 3, 0, 2, 0, 4, 3, 2, 3, 4, 2, 3, 3, 3, 4, 2, 0, 0, 4, 3, 2, 1,3, 1, 2, 4, 3, 2, 1, 2, 2, 4, 0, 1, 1, 4, 0, 1, 2, 3, 4, 4, 3, 3,2, 4, 3, 3, 0, 4, 1, 1, 0, 1, 0, 3],[1, 1, 3, 4, 3, 1, 1, 1, 3, 2, 0, 2, 3, 0, 0, 2, 2, 2, 0, 3, 4, 2,0, 1, 4, 3, 3, 4, 2, 4, 1, 0, 1, 3, 4, 2, 1, 0, 3, 0, 2, 3, 3, 4,3, 4, 4, 1, 0, 1, 2, 2, 1, 4, 4, 0, 3, 1, 2, 1, 2, 3, 2, 2, 0, 2,2, 1, 1, 3, 1, 3, 0, 4, 0, 3, 4, 0, 4, 2, 4, 2, 3, 1, 1, 4, 1, 0,3, 1, 1, 2, 0, 0, 0, 1, 2, 3, 4, 1],[3, 4, 3, 2, 0, 4, 0, 0, 2, 3, 1, 4, 0, 1, 0, 0, 3, 3, 0, 1, 1, 4,0, 2, 4, 2, 1, 3, 1, 1, 4, 0, 4, 0, 1, 2, 1, 3, 0, 2, 2, 1, 1, 1,3, 2, 1, 4, 1, 4, 2, 0, 3, 2, 1, 1, 0, 0, 3, 3, 0, 2, 2, 0, 4, 2,1, 3, 0, 2, 3, 3, 3, 0, 0, 1, 0, 1, 2, 1, 4, 3, 1, 0, 3, 3, 2, 1,0, 0, 2, 0, 1, 2, 4, 4, 0, 0, 1, 1],[3, 1, 4, 0, 0, 0, 3, 4, 3, 0, 1, 4, 2, 1, 4, 0, 1, 1, 2, 1, 3, 1,3, 4, 1, 0, 0, 1, 3, 1, 0, 1, 0, 1, 3, 0, 2, 4, 3, 0, 0, 0, 2, 3,4, 0, 4, 2, 3, 0, 3, 2, 1, 2, 1, 1, 3, 3, 3, 3, 3, 1, 2, 2, 0, 3,1, 0, 1, 2, 4, 1, 3, 3, 2, 3, 2, 1, 3, 1, 0, 1, 1, 3, 0, 4, 2, 0,1, 1, 1, 1, 3, 1, 4, 4, 1, 0, 4, 1],[1, 2, 2, 4, 3, 1, 3, 2, 2, 4, 0, 0, 1, 3, 4, 0, 4, 3, 3, 0, 2, 0,2, 1, 4, 4, 3, 2, 0, 3, 1, 1, 3, 0, 4, 3, 4, 0, 3, 1, 1, 0, 3, 0,0, 0, 3, 1, 3, 2, 0, 1, 4, 2, 3, 4, 0, 1, 2, 1, 3, 2, 3, 1, 3, 2,2, 0, 1, 2, 3, 1, 3, 3, 0, 1, 4, 0, 0, 2, 1, 3, 3, 4, 1, 1, 2, 4,3, 1, 1, 1, 2, 1, 4, 2, 0, 2, 2, 1]])
    y = np.array([1, 2, 3, 4, 5, 6])
    A =  [11, 23, 35, 26, 13, 0, 3, 2, 22, 4, 0, 0, 1, 3, 42, 0, 2, 3, 0, 2, 2, 3,4, 4, 2, 1, 4, 0, 0, 2, 3, 4, 3, 3, 4, 4, 2, 3, 3, 36, 52,21, 23, 0,0, 0, 3, 1, 3, 2, 0, 2,4, 5, 1, 2, 3, 4, 2, 1, 3, 2, 3, 1, 3, 2,2, 0, 1, 2, 3, 1, 3, 3, 0, 1, 4, 0, 0, 1, 2, 3, 4, 5, 6, 7, 5, 6,2, 1, 1, 1, 2, 1, 4, 2, 0, 2, 2, 1]
    '''
      A=[1, 2, 2, 4, 3, 1, 3, 2, 2, 4, 0, 0, 1, 3, 4, 0, 4, 3, 3, 0, 2, 0,
        2, 1, 4, 4, 3, 2, 0, 3, 1, 1, 3, 0, 4, 3, 4, 0, 3, 1, 1, 0, 3, 0,
        0, 0, 3, 1, 3, 2, 0, 1, 4, 2, 3, 4, 0, 1, 2, 1, 3, 2, 3, 1, 3, 2,
        2, 0, 1, 2, 3, 1, 3, 3, 0, 1, 4, 0, 0, 2, 1, 3, 3, 4, 1, 1, 2, 4,
        3, 1, 1, 1, 2, 1, 4, 2, 0, 2, 2, 1]
    '''
     # 朴素贝叶斯 
    NB = dmpy.supervised.naive_bayes.naive_bayes_learner()
    NB.train_nb(X,y)
    print NB.predict(np.array(A))    
    # SKlearn 朴素贝叶斯 
    clf = MultinomialNB()
    clf.fit(X, y)
    print(clf.predict(A))
    
     # 从文件中读取训练数据
    D, L = load()
    
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
