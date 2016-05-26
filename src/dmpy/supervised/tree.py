# -*- coding: utf-8 -*-
'''
@author: ren.j.j
e-mail:615989123@qq.com

训练数据表（样例）
---------------------------------------------------------------------------------------------------------------------------------------------------
      ------------------------------D-------------------------      ---------L------- 
key   age(col0)  inc(col1)         stu(col2)      cre(col3)          Class:buy?                 <---属性
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
import itertools

__all__ = [
    'decision_tree_learner',
    ]

class Node(object): 
    '''
    col        :   【int】                        属性分裂的序号 col = None代表叶子节点
    child      :   【一维数组 Node】               子树
    kind       :   【一维数组 int 】                对应child的kind
    Ci         :    [int]                        类标识
    '''
    def __init__(self, key):
        self.col     = key
        self.child   = np.array([],dtype = Node)
        self.Ci      = None 
    def add_child(self,child,kind):
        child.kind  = kind
        self.child    = np.insert(self.child , len(self.child),child)
       
def _info_D(L):
    '''     
       计算数据集信息熵  
     L  【一维数组】 存放类标L
    '''    
    info_D   = 0.0
    #   class下每个种类ci的个数【dict】
    kind_of_cis_count = Counter(L.tolist())
    kind_of_ci_count_sum = sum([ci_count[1]for ci_count in kind_of_cis_count.items()])
    
    # class下有多少种类ci【list】
    kind_of_cis= kind_of_cis_count.keys()
    for kind_of_ci in kind_of_cis:               
        info_D = info_D  + (-1)*(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)*sp.log2(kind_of_cis_count[kind_of_ci]/float(kind_of_ci_count_sum)))
    return info_D

def _gini_D(L):
    '''     
       计算数据集的基尼指数
     L  【一维数组】 存放类标L
    '''    
    gini_D   = 0.0
    #   class下每个种类ci的个数【dict】
    kind_of_cis_count = Counter(L.tolist())
    kind_of_ci_count_sum = sum([ci_count[1]for ci_count in kind_of_cis_count.items()])
    
    # class下有多少种类ci【list】
    kind_of_cis= kind_of_cis_count.keys()
    for kind_of_ci in kind_of_cis:       
        gini_D = gini_D  + sp.power(float(kind_of_cis_count[kind_of_ci])/kind_of_ci_count_sum,2)
        
    gini_D = 1 - gini_D
    return gini_D

def _info_col_D(cur_col,L):
    '''     
       计算列的信息熵  
       cur_col     
       L          【一维数组】 存放类标L
       cur_col    【list】    存放一列
    '''    
    assert len(cur_col) == len(L)
    
    info_col_D = 0.0    
    rows = len(cur_col)
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
            
        kovc_div_row = kind_of_values_count[kind_of_value]/float(rows)
        info_col_D = info_col_D + kovc_div_row * info_col_D_tmp
    return info_col_D

def _split_info_col_D(cur_col,L,Len_D):
    '''     
       计算列的分裂信息  
       cur_col     
       L          【一维数组】 存放类标L
       cur_col    【list】    存放一列
       Len_D       [int]     训练集元组的个数
    '''    
    assert len(cur_col) == len(L)
    
    split_info_col_D = 0.0    
    rows = len(cur_col)
    # col下每个种类values的个数【dict】
    kind_of_values_count = Counter(cur_col)
    # col下有多少种类values【list】
    kind_of_values= kind_of_values_count.keys()
    absolute_D = abs(Len_D)
    for kind_of_value in kind_of_values :
        # kind下包含value的个数
        absolute_Dj = abs(kind_of_values_count[kind_of_value])
        split_info_col_D = split_info_col_D+( float(absolute_Dj)/absolute_D * sp.log2(float(absolute_Dj)/absolute_D))
    split_info_col_D *= -1  
    return split_info_col_D

def _gini_col_D(cur_col,L):
    '''     
       计算列的基尼指数  
       cur_col     
       L               【一维数组】 存放类标L
       cur_col    【list】    存放一列
       
       返回值
       gini_col_D  当前列最小的基尼指数
       keys            二分之后属性值的keys
    '''    
    assert len(cur_col) == len(L)
    
    gini_col_D = float('Inf')  
    
    # col下每个种类values的个数【dict】
    kind_of_values_count = Counter(cur_col)
    kind_of_values_count_sum = sum([kind_of_value_count[1]for kind_of_value_count in kind_of_values_count.items()])
    abs_D  = kind_of_values_count_sum
    
    # col下有多少种类values【list】
    kind_of_values= kind_of_values_count.keys()

     #  二元划分集合
    two_partition_set = [] 
    two_partition_set_count  = (sp.power(2,len(kind_of_values))-2)/2
    for kind_of_value in range(1,len(kind_of_values)+1):
        iter = itertools.combinations(kind_of_values,kind_of_value)
        two_partition_set+=list(iter)
        if(len(two_partition_set) >= two_partition_set_count):
            break ;
        
    if two_partition_set_count == 1 :
        two_partition_set.pop()
    elif two_partition_set_count == 0 :
         two_partition_set_count+=1
        
    assert len(two_partition_set) == two_partition_set_count
    
    #  在二元划分集合找到最小的分裂子集
    for a_two_partition_set in two_partition_set:
        Class  = []
        keys   = []
        abs_D1 = 0
        for a_tup in a_two_partition_set:
            abs_D1 +=  abs(kind_of_values_count[a_tup])
            #  value对应的key 
            keys= [key for key,value in enumerate(cur_col) if value == a_tup]
            #  key对应的class  
            Class =  [L[i] for i in keys]
        gini_D1 = _gini_D(np.array(Class))
        D1_DIV_D_MULTI_GINID1 = float(abs_D1)/abs_D*gini_D1
         
        abs_D2 = abs_D - abs_D1
        Class = [L[i] for i in list(set(range(0,kind_of_values_count_sum))-set(keys))]
        gini_D2 = _gini_D(np.array(Class))
        D2_DIV_D_MULTI_GINID2 = float(abs_D2)/abs_D*gini_D2
         
        if gini_col_D > D1_DIV_D_MULTI_GINID1 + D2_DIV_D_MULTI_GINID2:
            gini_col_D = D1_DIV_D_MULTI_GINID1 + D2_DIV_D_MULTI_GINID2
            
    return gini_col_D , keys , list(set(range(0,kind_of_values_count_sum))-set(keys))
     
def _split_C45(D , D_col,L , Len_D):
    '''
     D          【二维数组】 存放记录D
     D_col      【一维数组】 存放记录D的列名        
     L          【一维数组】 存放类标L
     Len_D       [int]     训练集元组的个数
    '''    
    assert D.shape[1] == len(D_col)
    
    #   最大增益率
    max_gain_rate =  float('-Inf')  
    #   数据集的信息熵
    info_D   = 0.0
    #   最好的分裂属性  
    best_col = None
    
    #   计算数据集信息熵  
    info_D =   _info_D(L)      
    #  计算所有列的信息熵，      
    rows,cols = D.shape
    for col in range(cols) :
        #  当前列
        cur_col = D[:,col]
        #  当前列的分裂信息
        split_info_col_D   = _split_info_col_D(cur_col,L,Len_D)
        #  当前列的信息熵
        info_col_D         = _info_col_D(cur_col,L)
        #  当前列的信息增益
        gain_col_D         = info_D - info_col_D
        #  当前列的增益率
        gain_rate_col_D    = gain_col_D / split_info_col_D
        
        # 获得最大的信息增益
        if  gain_rate_col_D >  max_gain_rate :
            max_gain_rate = gain_rate_col_D
            best_col = D_col[col]            
            
    return  best_col

def _split_ID3(D , D_col, L):
    '''
     D      【二维数组】 存放记录D     
     D_col  【一维数组】 存放记录D的列名      
     L      【一维数组】 存放类标L
    '''    
    assert D.shape[1] == len(D_col)
    #   最大信息增益
    max_gain =  float('-Inf')  
    #   数据集的信息熵
    info_D   = 0.0
    #  最好的分裂属性  
    best_col = None
    
    # 计算数据集信息熵  
    info_D =   _info_D(L)
    
    
    #  计算所有列的信息熵，      
    rows,cols = D.shape
    for col in range(cols) :
          # 当前列的信息熵
        cur_col = D[:,col]
        info_col_D = _info_col_D(cur_col,L)
          # 获得最大的信息增益
        if  info_D - info_col_D >  max_gain :
            max_gain = info_D - info_col_D
            best_col = D_col[col]          
    return  best_col

def _split_GINI(D , L):
    '''
    D  【二维数组】 存放记录D      
    L  【一维数组】 存放类标L
    '''    
    #   最大信息增益
    max_gini =  float('-Inf')  
    #   数据集的基尼指数
    gini_D    = 0.0
    keys_D1 = []
    keys_D2 = []
    #  最好的分裂属性  
    best_col = None
    # 计算数据集信息熵  
    gini_D =   _gini_D(L)
     
    #  计算所有列的基尼指数，      
    rows,cols = D.shape
    for col in range(cols) :
        # 当前列的基尼指数
        cur_col = D[:,col]
        gini_col_D , keysD1 ,keysD2= _gini_col_D(cur_col,L)
        # 获得最大的信息增益
        if  gini_D - gini_col_D >  max_gini :
            max_gini = gini_D - gini_col_D
            keys_D1  = keysD1
            keys_D2  = keysD2
            best_col = col          
    return  best_col,keys_D1,keys_D2
 
def _build_tree(D,Len_D ,D_col ,L, criterion, min_split, multiple_branch):
    '''
     D                 【二维数组】 存放记录    
     Len_D              [int]     训练集元组初始的总个数  
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
    if criterion   == 'ID3' :
        best_col  = _split_ID3(D, D_col ,L) 
    elif criterion == 'C45' :
        best_col  = _split_C45(D,D_col, L ,Len_D) 
    elif criterion == 'CART' :
        best_col ,keys_D1,keys_D2 = _split_GINI(D, L) 
        N.col  = best_col
        # CART产生二叉树，与ID3和C4.5的分区划分略有不同
        Dj= D[keys_D1]
        Lj= L[keys_D1]
        if len(Dj) == 0  :
                #  加入一个树叶到节点N，标记D为多数类
            max_ci_count = 0
            for i in L :
                if(max_ci_count<L.tolist().count(i)):
                    max_ci_count = L.tolist().count(i)
                    N.ci         = i 
            return N
        else:
            child = _build_tree(Dj,Len_D ,D_col ,Lj, criterion, min_split, multiple_branch)    
            kind_of_best_col_value = list(set(Dj[:,best_col]))
            N.add_child(child ,kind_of_best_col_value)          
        Dj= D[keys_D2]
        Lj= L[keys_D2]
        if len(Dj) == 0  :
                #  加入一个树叶到节点N，标记D为多数类
            max_ci_count = 0
            for i in L :
                if(max_ci_count<L.tolist().count(i)):
                    max_ci_count = L.tolist().count(i)
                    N.ci         = i 
            return N
        else:
            child = _build_tree(Dj, Len_D,D_col ,Lj, criterion, min_split, multiple_branch)    
            kind_of_best_col_value = list(set(Dj[:,best_col]))
            N.add_child(child ,kind_of_best_col_value)
        return N;
        
        
    #  用最好的分裂属性标记节点N
    N.col  = best_col

                        
    #  划分元组输出Dj、Lj并且对每个分区产生子树
    best_col_values = D[:,D_col.index(best_col)]
    
     #  如果允许多路划分，删除分裂的属性
    if multiple_branch :
        D = np.delete(D, D_col.index(best_col), axis = 1) 
        D_col.remove(best_col)  
            
    #  分裂属性下每个种类values的个数【dict】
    kind_of_best_col_values_count = Counter(best_col_values)
    #  分裂属性下有多少种类values【list】
    kind_of_best_col_values= kind_of_best_col_values_count.keys()
    for kind_of_best_col_value in kind_of_best_col_values :      
        #  value对应的key 
        keys= [key for key,value in enumerate(best_col_values) if value == kind_of_best_col_value]
        Dj= D[keys]
        Lj= L[keys]
                
        if Dj.shape[1] == 0 or Dj.shape[0] == 0 :
            #  加入一个树叶到节点N，标记D为多数类
            max_ci_count = 0
            for i in L :
                if(max_ci_count<L.tolist().count(i)):
                    max_ci_count = L.tolist().count(i)
                    N.ci         = i 
            return N
        else:
            child = _build_tree(Dj,Len_D ,D_col[:] ,Lj, criterion, min_split, multiple_branch)    
            N.add_child(child , [kind_of_best_col_value])
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
        str1 = '\n%s '   % (ci_name[node.Ci])
        for kind_item in node.kind:
            str_kind_item = "%i " %kind_item
            str1=str_kind_item+str1
        plt.annotate(str1, xy = (width[depth-1]*width_offset+5,depth*depth_offset), fontsize=8) 
        if(location != [0,0] ):
            ax.plot([width[depth-1]*width_offset , location[1]*width_offset]  , [depth*depth_offset,location[0]*depth_offset], 'b-',linewidth=.1)
      # 打印分裂节点
    else :  
        ax.plot(width[depth-1]*width_offset,depth*depth_offset,  'ro')
        str1 =''
        if hasattr(node,'kind') :
            str1 = '\n%s'   %  (list_name[node.col])         
            for kind_item in node.kind:
                str_kind_item = "%i " %kind_item
                str1=str_kind_item+str1
        else:
            str1 = '%s '   %  (list_name[node.col] ) 
        plt.annotate(str1, xy = (width[depth-1]*width_offset+5,depth*depth_offset), fontsize=8) 
        if(location != [0,0] ):
          ax.plot([width[depth-1]*width_offset , location[1]*width_offset]  , [depth*depth_offset,location[0]*depth_offset], 'b-',linewidth=.1)
       
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
    for node in tree.child :
        if node is tree.child[0]:
            d = d+1         
            if depth[0] < d :
                depth[0] = d
        _tree_w_d(node,d,width,depth)
        width[d-1]= width[d-1]+1
        
def _apply_tree(tree, features):
    '''
    应用决策树
    conf = apply_tree(tree, features)
    features  :    【一维数组 int 】     一个需要预测的数据元组      
    '''
    if tree.col is None:
        return tree.Ci
    
    features_list = list(features)
    features_col_value = features[tree.col]
    for child in tree.child :
        if(child.kind == features_col_value):
            return _apply_tree(child,features)

      #  一个数据类型错误的预测数据元组，该列值在训练元组中相应的列上不存在
    assert False
                                        
class decision_tree_learner(object):

    def __init__(self,criterion='ID3', min_split=4,multiple_branch=True):
        '''
         criterion           【string】属性选择划分的度量 默认是ID3（信息增益）
         min_split           【int】        停止分裂的数据集记录数
         multiple_branch     【bool】     是否多路划分
        '''     
        self.criterion       = criterion
        self.min_split       = min_split
        self.multiple_branch = multiple_branch    
           #  当前决策树的深度
        self.depth           = [0] 
           #  当前决策树的宽度
        self.width           = [0]
        
    def train_tree(self,D, L, weights=None):
        '''
        D    【二维数组】 存放记录      
        L    【一维数组】 存放类标志ci（i=1..n）        
        criterion 【函数指针】 属性选择划分的度量
        '''
        r,c = D.shape
        D_col =np.arange(c).tolist()  
        tree = _build_tree(D,r,D_col, L, self.criterion, self.min_split, self.multiple_branch)
        return tree_model(tree)
            
class tree_model():
     '''
    tree model
    '''
     def __init__(self, tree):
        self.tree = tree

     def apply(self,feats):
        return _apply_tree(self.tree, feats)
    
     def tree_w_d(self,tree):
        '''
        获得生成决策树的宽度和深度
        '''
        depth =[1]
        width =[1]
        _tree_w_d(tree,1,width,depth)
        return [max(width),depth[0]] 
    
     def print_tree(self,list_name =[], ci_name=[],width_offset = 100 , depth_offset = 50 ,outfile = '/test.png'):
        '''
        list_name          属性列表的lis名称t映射
        ci_name            状态列表的lis名称t映射
        width_offset       在行上宽度的偏移量
        depth_offset       在列上深度的偏移量
        apply_data_class   预测的数据结果，在图形上进行描红
        '''   
           # 画布
        fig = plt.figure(figsize=(13,13))
        

           # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
        ax = fig.add_subplot(111)
           # 坐标
        ax.grid()
        width ,depth = self.tree_w_d(self.tree )
        l=[0,(width+1)*width_offset,(depth+1)*depth_offset,0]  
        plt.axis(l)

           # 打印树  
        depth =1
        width =[1]
        _print_tree(self.tree ,ax,width_offset,depth_offset,list_name,ci_name,1,width,[0,0])  
           # 保存成图片
        plt.show()  
        plt.savefig(outfile, dpi=240)
tree_model