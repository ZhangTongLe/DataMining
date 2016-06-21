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


def print_Livermore(datafile,start_date,end_date,key_value,second_key_value) :
    
    datafile111 = os.path.dirname(__file__)+'/'+datafile+'.csv'
    _winedatafile = datafile111
    data1  = np.array([list(map(float,line.split(','))) for line in open(_winedatafile)])
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
            if i[0] <= start_date :
                s_sub = sub+1
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

    table  = []     
    c_1_v     = []# 次级上升
    c_2_v     = []# 自然上升
    c_3_v     = []# 上升趋势
    c_4_v     = []# 下降趋势
    c_5_v     = []# 自然下降
    c_6_v     = []# 次级下降
    c_7_v     = []
    c_1_d     = []
    c_2_d     = []
    c_3_d     = []
    c_4_d     = []
    c_5_d     = []
    c_6_d     = []
    c_7_d     = []
    prv_r     = None 
    prv_col   = None
    cur_col   = None
    for cur_r in range(0, D_L.__len__())[::-1]:
        sub = cur_r
        cur_r = D_L[cur_r]
        if prv_r is None :
            prv_r = cur_r 
            continue
        
          
        if   cur_r[1] > prv_r[1]: # 上升趋势
            if len(c_4_v) >0 :
               if(cur_r[1]- key_value <  min(c_4_v)): # 是自然上升么？
                   cur_col = 4                        # 不是 是下降
               else:
                   cur_col = 2                        # 是 是自然上升
            elif len(c_5_v) >0 :
               if(cur_r[1]- key_value <  min(c_5_v)): # 是自然上升么？
                   cur_col = 5                        # 不是 是下降
               else:
                   cur_col = 2                        # 是 是自然上升
            elif  len(c_2_v) >0 :
                   for i_sub in range(0, table.__len__())[::-1]:
                      i=table[i_sub]
                      if i[0] == 2 or i[0] == 3 :
                        if(cur_r[1]+ second_key_value> max(i[2])):
                             cur_col = 3
                             break
                   if i_sub == 0:
                       cur_col = 2  
            else:
                   cur_col = 3                        # 上升趋势
        elif cur_r[1] < prv_r[1]: # 下降趋势
            if len(c_3_v) >0 :
               if(cur_r[1]+ key_value >  max(c_3_v)): # 是自然下降么？
                   cur_col = 3                        # 不是 是上升
               else:
                   cur_col = 5                        # 是 是自然下降
            elif  len(c_2_v) >0 :
               if(cur_r[1]+ key_value >  max(c_2_v)): # 是自然下降么？
                   cur_col = 2                        # 不是 是上升
               else:
                   cur_col = 5                        # 是 是自然下降
            elif  len(c_5_v) >0 :
                   for i_sub in range(0, table.__len__())[::-1]:
                      i=table[i_sub]
                      if i[0] == 4 or i[0] == 5 :
                        if(cur_r[1] - second_key_value< min(i[2])):
                             cur_col = 4
                             break
                   if i_sub == 0:
                      cur_col = 5  
            else:
                   cur_col = 4                        # 下降趋势
            
        if cur_col != prv_col or sub == 0:
            if prv_col == 3 :   
                if cur_col == 4 :
                    c_4_v  = c_4_v + [c_3_v[len(c_3_v)-1]]  #保证曲线完整
                    c_4_d  = c_4_d + [c_3_d[len(c_3_d)-1]]    
                if cur_col == 5 :
                    c_5_v  = c_5_v + [c_3_v[len(c_3_v)-1]]
                    c_5_d  = c_5_d + [c_3_d[len(c_3_d)-1]]                                                     
                t_r =[3,c_3_d,c_3_v]
                table = table + [t_r]
                c_3_v  = []
                c_3_d  = []
            elif prv_col == 4 :
                if cur_col == 3 :
                    c_3_v  = c_3_v + [c_4_v[len(c_4_v)-1]]
                    c_3_d  = c_3_d + [c_4_d[len(c_4_d)-1]]    
                if cur_col == 2 :
                    c_2_v  = c_2_v + [c_4_v[len(c_4_v)-1]]
                    c_2_d  = c_2_d + [c_4_d[len(c_4_d)-1]]    
                t_r =[4,c_4_d,c_4_v]
                table = table + [t_r]
                c_4_v  = []
                c_4_d  = []
            elif prv_col == 2 :
                if cur_col == 4 :
                    c_4_v  = c_4_v + [c_2_v[len(c_2_v)-1]]
                    c_4_d  = c_4_d + [c_2_d[len(c_2_d)-1]]    
                if cur_col == 5 :
                    c_5_v  = c_5_v + [c_2_v[len(c_2_v)-1]]
                    c_5_d  = c_5_d + [c_2_d[len(c_2_d)-1]]
                if cur_col == 3 :
                    c_3_v  = c_3_v + [c_2_v[len(c_2_v)-1]]
                    c_3_d  = c_3_d + [c_2_d[len(c_2_d)-1]]
                t_r =[2,c_2_d,c_2_v]
                table = table + [t_r]
                c_2_v  = []
                c_2_d  = []
            elif prv_col == 5 :
                if cur_col == 3 :
                    c_3_v  = c_3_v + [c_5_v[len(c_5_v)-1]]
                    c_3_d  = c_3_d + [c_5_d[len(c_5_d)-1]]    
                if cur_col == 2 :
                    c_2_v  = c_2_v + [c_5_v[len(c_5_v)-1]]
                    c_2_d  = c_2_d + [c_5_d[len(c_5_d)-1]]   
                if cur_col == 4 :
                    c_4_v  = c_4_v + [c_5_v[len(c_5_v)-1]]
                    c_4_d  = c_4_d + [c_5_d[len(c_5_d)-1]]    
                t_r =[5,c_5_d,c_5_v]
                table = table + [t_r]
                c_5_v  = []
                c_5_d  = []
        
        str = repr(cur_r[0])
        year  = int(str[0])*1000+int(str[1])*100+int(str[2])*10+ int(str[3])
        month = int(str[4])*10+int(str[5])
        day   = int(str[6])*10+int(str[7])
        date  = datetime(year, month, day, 0, 0, 0,0)
        
        if cur_col   == 3 :
            c_3_v  = c_3_v+[cur_r[1]]
            c_3_d  = c_3_d+[sub]
        elif cur_col == 4 :
            c_4_v  = c_4_v+[cur_r[1]]
            c_4_d  = c_4_d+[sub]
        elif cur_col == 2 :
            c_2_v  = c_2_v+[cur_r[1]]
            c_2_d  = c_2_d+[sub]
        elif cur_col == 5 :
            c_5_v  = c_5_v+[cur_r[1]]
            c_5_d  = c_5_d+[sub]
           
        prv_col  = cur_col
        prv_r    = cur_r
                
     # 画布
    fig = plt.figure(figsize=(39,13),facecolor=(1, 1, 0))       
      # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax = fig.add_subplot(111,axisbg='w')
      # 坐标
    ax.grid()
    
    
    timespan = timedelta(days=1)
    D_max = max(D)+timespan
    D_min = min(D)-timespan
    L_max = max(L)+10
    L_min = min(L)-10
    
     
    l=[len(D),0,L_min,L_max]  
    plt.axis(l)

     # 1次级上升
     # 2 自然上升
     # 3上升趋势
     # 4下降趋势
     # 5自然下降
     # 6次级下降
    for i in table :
        if   i[0] == 3 :
            ax.plot(i[1],i[2],'r-')
            y = max(i[2])
            x = i[1][i[2].index(y)]
            time = datetime.strftime(D[x], '%m%d')
            str1 = '%s\n%d'% (time,y) 
            plt.annotate(str1, xy = (x,y), fontsize=12,  color="red") 
        elif i[0] == 4 :
            ax.plot(i[1],i[2],'g-')
            y = min(i[2])
            x = i[1][i[2].index(y)]
            time = datetime.strftime(D[x], '%m%d')
            str1 = '%s\n%d'% (time,y) 
            plt.annotate(str1, xy = (x,y), fontsize=12,  color="green") 
        elif i[0] == 2 :
            ax.plot(i[1],i[2],'b-')
            y = max(i[2])
            x = i[1][i[2].index(y)]
            time = datetime.strftime(D[x], '%m%d')
            str1 = '%s\n%d'% (time,y) 
            plt.annotate(str1, xy = (x,y), fontsize=12,  color="blue") 
        elif i[0] == 5 :
            ax.plot(i[1],i[2],'c-')
            y = min(i[2])
            x = i[1][i[2].index(y)]
            time = datetime.strftime(D[x], '%m%d')
            str1 = '%s\n%d'% (time,y) 
            plt.annotate(str1, xy = (x,y), fontsize=12,  color='c') 
    
      
     # 保存成图片
    plt.show()  
    time ="_%i_to_%i"%(start_date,end_date)
    datafile111 = os.path.dirname(__file__)+'/'+datafile+time+'.png'
    plt.savefig(datafile111, dpi=240,facecolor=(1, 1, 1))

if __name__ == '__main__' or __name__ == 'test_tree':    
    print_Livermore('BLC',20140306,20140613,25,10) 
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
