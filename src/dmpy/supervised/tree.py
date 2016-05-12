# -*- coding: utf-8 -*-
'''
@author: ren.j.j
e-mail:615989123@qq.com

训练数据表（样例）
---------------------------------------------------------------------------------------------------------------------------------------------------
      ------------------------------D-------------------------      ---------L------- 
key   age(col1)  inc(col2)         stu(col3)      cre(col4)          Class:buy?                 <---属性
0       1,(value)      1,             1,             1                  1(ci)                   <--- row1 
1       1,(value)      1,             1,             2                  1                       <--- row2 
2       2,             1,             1,             1                  0                       <--- row3 
3       3,             2,             1,             1                  0                       <--- row4 
4       3,             3,             2,             1                  0                       <--- row5 
5       3,             3,             2,             2                  1                       <--- row6 
6       2,             3,             2,             2                  0                       <--- row7 
7       1,(value)      2,             1,             1                  1                       <--- row8 
8       1,(value)      3,             2,             1                  0                       <--- row9 
9       3,             2,             2,             1                  0                       <--- row10 
10      1,(value)      2,             2,             2                  0                       <--- row11 
11      2,             2,             1,             2                  0                       <--- row12
12      2,             1,             2,             1                  0                       <--- row13 
13      3,             2,             1,             2                  1                       <--- row14 
---------------------------------------------------------------------------------------------------------------------------------------------------
age(col1)下值（value）对应key=0,1,7,8,10 属于同一种kind
---------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import scipy as sp
from collections import Counter
import matplotlib.pyplot as plt


__all__ = [
    'decision_tree_learner',
    ]

class Node(object): 
    '''
    col        :   【int】                       属性分裂的序号 col = None代表叶子节点
    child    :   【一维数组 Node】   子树
    kind     :   【一维数组 int 】       对应child的kind
    Ci         :     [int]     
    '''
    def __init__(self, key):
        self.col     = key
        self.child   = np.array([],dtype = Node)
        #self.kind    = np.array([])
        self.Ci      = None 
    def add_child(self,child,kind):
        child.kind  = kind
        self.child    = np.insert(self.child , len(self.child),child)
        #self.kind  = np.insert(self.kind  , len(self.kind) ,kind)
        
def _split_ID3(D , L):
    '''
     D  【二维数组】 存放记录D      
     L  【一维数组】 存放类标L
    '''
    
    #   最大信息增益
    max_gain =  float('-Inf')  
    #   数据集的信息熵
    info_D   = 0.0
    #  最好的分裂属性  
    best_col = None
    
    #   计算数据集信息熵  
    #   class下每个种类ci的个数【dict】
    kind_of_cis_count = Counter(L.tolist())
    kind_of_ci_count_sum = sum([ci_count[1]for ci_count in kind_of_cis_count.items()])
    
    # class下有多少种类ci【list】
    kind_of_cis= kind_of_cis_count.keys()
    for kind_of_ci in kind_of_cis:               
        info_D = info_D  + (-1)*(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)*sp.log2(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)))
          
    #  计算所有列的信息熵，      
    rows,cols = D.shape
    for col in range(cols) :
            #当前列的信息熵
        info_col_D = 0.0        
        cur_col = D[:,col]
        
        # col下每个种类values的个数【dict】
        kind_of_values_count = Counter(cur_col)
        # col下有多少种类values【list】
        kind_of_values= kind_of_values_count.keys()
        for kind_of_value in kind_of_values :
            #  value对应的key 
            keys= [key for key,value in enumerate(cur_col) if value == kind_of_value]
            #  key对应的class  
            Class =  [L[i] for i in keys]
            #  class下每个种类ci的个数【dict】
            kind_of_cis_count = Counter(Class)
            kind_of_ci_count_sum = sum([ci_count[1]for ci_count in kind_of_cis_count.items()])
            # class下有多少种类ci【list】
            kind_of_cis= kind_of_cis_count.keys()
                #  逐项计算累加
            info_col_D_tmp = .0
            for kind_of_ci in kind_of_cis:               
                info_col_D_tmp = info_col_D_tmp  +(-1)*(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)*sp.log2(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)))
            
            #  kind_of_values_count = kovc
            kovc_div_row = kind_of_values_count[kind_of_value]/float(rows)
            info_col_D = info_col_D + kovc_div_row * info_col_D_tmp
            
            # 获得最大的信息增益
        if  info_D - info_col_D >  max_gain :
            max_gain = info_D - info_col_D
            best_col = col        
    return  best_col

def _build_tree(D, D_col ,L, criterion, min_split, multiple_branch):
    '''
     D                 【二维数组】 存放记录      
     D_col             【一维数组】 属性列表      
     L                 【一维数组】 存放类标志Ci（i=0..n）    
     criterion         【函数指针】 属性选择划分的度量
     min_split         【int】     停止分裂的数据集记录数
     multiple_branch   【bool】    是否多路划分
    '''    
    # 记录和类标志的长度匹配
    assert len(D) == len(L)

    
    #  创建一个节点N
    N = Node(None)
    
 
    #  D中的元组都属于同一类,返回N作为叶子节点，用Ci标记
    if np.max(L) == np.min(L):
        N.Ci  = L[0]
        return N 

    #  D中的属性为空,返回N作为叶节点，标记Ci中的多数类
    if len(D_col) == 0:
        max_ci_count = 0
        for i in L :
            if(max_ci_count<L.tolist().count(i)):
                max_ci_count = L.tolist().count(i)
                N.ci         = i 
        return N
        
    #  找出最好的分裂属性
    best_col  = None
    if criterion == 'ID3' :
        best_col  = _split_ID3(D, L) 
        
    #  用最好的分裂属性标记节点N
    N.col  = best_col
    
    #  划分元组输出Dj、Lj并且对每个分区产生子树
    best_col_values = D[:,best_col]
    #  分裂属性下每个种类values的个数【dict】
    kind_of_best_col_values_count = Counter(best_col_values)
    #  分裂属性下有多少种类values【list】
    kind_of_best_col_values= kind_of_best_col_values_count.keys()
    for kind_of_best_col_value in kind_of_best_col_values :
        #  value对应的key 
        keys= [key for key,value in enumerate(best_col_values) if value == kind_of_best_col_value]
        Dj= D[keys]
        Lj= L[keys]
        #  如果允许多路划分，删除分裂的属性
        if multiple_branch :
            Dj = np.delete(Dj, best_col, axis = 1)
        
        if len(Dj) == 0  :
            #  加入一个树叶到节点N，标记D为多数类
            max_ci_count = 0
            for i in L :
                if(max_ci_count<L.tolist().count(i)):
                    max_ci_count = L.tolist().count(i)
                    N.ci         = i 
            return N
        else:
            child = _build_tree(Dj, D_col ,Lj, criterion, min_split, multiple_branch)    
            N.add_child(child ,kind_of_best_col_value)
    return N;

def _print_tree(node ,ax,width_offset,depth_offset,list_name,ci_name,depth,width,location):
    '''
     node               存放记录    决策树      
     width_offset       在行上宽度的偏移量
     depth_offset       在列上深度的偏移量
     list_name          属性列表的lis名称t映射
     ci_name            状态列表的lis名称t映射
     depth              【int】 当前深度
     width              【list】每一行的宽度
     location            上层父节点的位置
    '''    
    if(depth > len(width)):
        width += [1]
    # 打印叶子节点  
    if node.col == None:
        ax.plot(width[depth-1]*width_offset,depth*depth_offset, 'bo')
        str1 = '%s  (%d)'   % (ci_name[node.Ci] , node.kind )
        plt.annotate(str1, xy = (width[depth-1]*width_offset,depth*depth_offset)) 
        if(location != [0,0] ):
            ax.plot([width[depth-1]*width_offset , location[1]*width_offset]  , [depth*depth_offset,location[0]*depth_offset], 'b-')
    # 打印分裂节点
    else :  
        ax.plot(width[depth-1]*width_offset,depth*depth_offset,  'ro')
        str1 =''
        if hasattr(node,'kind') :
            kind = node.kind
            str1 = '%s  (%d)'   %  (list_name[node.col],  kind ) 
        else:
            str1 = '%s '   %  (list_name[node.col] ) 
        plt.annotate(str1, xy = (width[depth-1]*width_offset,depth*depth_offset)) 
        if(location != [0,0] ):
            ax.plot([width[depth-1]*width_offset , location[1]*width_offset]  , [depth*depth_offset,location[0]*depth_offset], 'b-')
    location = [depth ,width[depth-1] ]
    for child in node.child :
            if child is node.child[0]:
                depth= depth +1        
            _print_tree(child ,ax,width_offset,depth_offset,list_name,ci_name,depth,width,location)
            width[depth-1]= width[depth-1]+1


def _tree_w_d(tree,d,width,depth):
    '''
      获得树的宽度和深度
     d                    【int】  初始树的深度 1      
     width             【list】 树的所有层的节点个数   
     depth             【list】 树的最大宽度
     '''
    if(d > len(width)):
        width += [0]
    #print d ,width
    for node in tree.child :
        if node is tree.child[0]:
            d = d+1         
            if depth[0] < d :
                depth[0] = d
        _tree_w_d(node,d,width,depth)
        width[d-1]= width[d-1]+1
        
class decision_tree_learner(object):
    '''
    criterion           【string】属性选择划分的度量 默认是ID3（信息增益）
    min_split           【int】        停止分裂的数据集记录数
    multiple_branch     【bool】     是否多路划分
    '''
    def __init__(self,criterion='ID3', min_split=4,multiple_branch=True):
        self.criterion       = criterion
        self.min_split       = min_split
        self.multiple_branch = multiple_branch    
           #  当前决策树的深度
        self.depth           = [0] 
           #  当前决策树的宽度
        self.width           = [0]
        
    '''
    D    【二维数组】 存放记录      
    L    【一维数组】 存放类标志ci（i=1..n）        
    criterion 【函数指针】 属性选择划分的度量
    '''
    def train_tree(self,D, L, weights=None):
        r,c = D.shape
        D_col =np.arange(c)  
        tree = _build_tree(D,D_col, L, self.criterion, self.min_split, self.multiple_branch)
        return tree
    
    def tree_w_d(self,tree):
        depth =[1]
        width =[1]
        _tree_w_d(tree,1,width,depth)
        return [max(width),depth[0]] 
    
        '''
        root               决策树根节点      
        list_name          属性列表的lis名称t映射
        ci_name            状态列表的lis名称t映射
        width_offset       在行上宽度的偏移量
        depth_offset       在列上深度的偏移量
        '''   
    def print_tree(self,root,list_name =[], ci_name=[],width_offset = 100 , depth_offset = 100 ):
           # 画布
        fig = plt.figure(figsize=(13,13))
           # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
        ax = fig.add_subplot(111)
           # 坐标
        ax.grid()
        width ,depth = self.tree_w_d(root)
        l=[0,(width+1)*width_offset,(depth+1)*depth_offset,0]  
        plt.axis(l)
           # 打印树  
        depth =1
        width =[1]
        _print_tree(root,ax,width_offset,depth_offset,list_name,ci_name,1,width,[0,0])  
           # 保存成图片
        plt.show()  
        plt.savefig('/test.png', dpi=120)
