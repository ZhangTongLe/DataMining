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
    col        :   【int】                         属性分裂的序号 col = None代表叶子节点
    child      :   【一维数组 Node】                 子树
    kind       :   【一维数组 int 】                 对应child的kind
    Ci         :    [int]                         类标识
    Ci_count   :    元组                               当前节点的Ci分类个数统计                
    '''
    def __init__(self, key):
        self.col     = key
        self.child   = np.array([],dtype = Node)
        self.Ci      = None 
        self.Ci_count= None
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
    
    N.Ci_count = Counter(L)
 
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
                N.Ci         = i 
        return N
        
      # 一个节点下的样本属性太少  
    if D.shape[0] < min_split:
        #  加入一个树叶到节点N，标记D为多数类
      max_ci_count = 0
      for i in L:
            if(max_ci_count<L.tolist().count(i)):
                max_ci_count = L.tolist().count(i)
                N.Ci         = i 
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
        child = _build_tree(Dj,Len_D ,D_col ,Lj, criterion, min_split, multiple_branch)    
        kind_of_best_col_value = list(set(Dj[:,best_col]))
        N.add_child(child ,kind_of_best_col_value)          
        
        Dj= D[keys_D2]
        Lj= L[keys_D2]
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
        if([features_col_value] in child.kind ):
            return _apply_tree(child,features)

      #  一个数据类型错误的预测数据元组，该列值在训练元组中相应的列上不存在
    assert False
     
def _apply_error(node ,D, L):  
    '''
        训练误差（再代入误差），使用训练集作为样本，是种乐观估计，训练集好的时候可以作为泛化误差
        D                 【二维数组】 存放记录    
        L                 【一维数组】 存放类标志Ci（i=0..n）   
              返回值 ： 误差率、误差个数
    '''         
    assert len(D) == len(L)
    result_l = [_apply_tree(node,Di) for Di in D]
    assert len(result_l) == len(L.tolist())
    result_compare =  np.transpose(np.array((L,result_l))) 
    result_compare  = [x[0]==x[1] for x in result_compare]   
     # True  = 1 ,False = 0   
    result_right = np.count_nonzero(result_compare)   
    return 1 -(result_right/float(len(result_compare))),len(result_compare) - result_right       
               
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
           '''
            应用决策树
            features  :    【一维数组 int 】     一个需要预测的数据元组      
         '''
           return _apply_tree(self.tree, feats)
    
     def apply_error(self,D, L):
           '''
              训练误差（再代入误差），使用训练集作为样本，是种乐观估计，训练集好的时候可以作为泛化误差
               D                 【二维数组】 存放记录    
               L                 【一维数组】 存放类标志Ci（i=0..n）   
              返回值 ： 误差率、误差个数
          '''    
           return _apply_error(self.tree ,D, L)
    
     def pessimistic_error(self,D,L,penalty_term):
           '''
              悲观误差评估 --- 该评估结合了模型复杂度（叶子节点的个数）
               D                 【二维数组】 存放记录    
               L                 【一维数组】 存放类标志Ci（i=0..n）   
               penalty_term      【复杂度罚项】 
          '''   
           assert len(D) == len(L)
              # 决策树总的训练误差
           apply_error_rate , e_T_= self.apply_error(D, L)
           
              # 决策书叶子节点个数
           def recursion(node , leaf_count):
              if(node.col == None) :
                leaf_count[0] = leaf_count[0] + 1
              for child in node.child:
                recursion(child,leaf_count)
           leaf_count=[0]
           recursion(self.tree,leaf_count)
             
              # 罚项总和
           omega_T  = leaf_count[0]*penalty_term
           
              #  训练记录个数
           N_t = len(L)
           
           return (e_T_+omega_T)/float(N_t)
       
     def PEP(self,D, L, penalty_term = 0.5):
           '''
                  悲观误差剪枝 --- Pessimistic Error Pruning(PEP，悲观剪枝）
                Quinlan提出的PEP剪枝算法是通过比较剪枝前和剪枝后的错分样本数来判断是否剪枝的它在ID3系统中获得实现。PEP既采用训练集来生成决策树又用它来进行剪枝,
              不需要独立的剪枝集。这样,由于决策树生成和剪枝都使用训练集,所以产生的错分样本率r(t)是有偏差的,即偏向于训练集,无法得到最优的剪枝树。对此Quinlan引入了一
              个基于二项分布的连续校正公式来对在训练集中产生的错分样本率r(t)进行校正,经过校正后可以得到一个较为合理的错分样本率。
                  假设对于训练集的错分样本率为r(t),计算公式为r(t)=e(t)/n(t);
                  连续校正后的错分样本率为r′(t),计算公式为r′(t)=[e(t)+1/2]/n(t)。
                  为简单起见,在下面计算过程中采用错分样本数而不用错分样本率来说明问题。经过连续校正后,对节点t进行剪枝产生的错分样本数为e′(t)=[e(t)+1/2];未剪
              枝的错分样本数变为e′(Tt)=∑[e(s)+1/2],s∈{Tt子树的所有叶节点};进一步,引入子树Tt的服从二项分布的标准错误
            SE(e′(Tt)),SE(e′(Tt))=[e′(Tt)・(n(t)-e′(Tt))/n(t)]^(1/2)。
                PEP算法采用自顶向下的顺序遍历完全决策树Tmax,对于每个内部节点t逐一比较e′(t)和e′(Tt)+SE(e′(Tt))的大小,当满足条件e′(t)<=e′(Tt)+SE(e′(Tt))时,
              就进行剪枝,剪掉以t为根节点的子树Tt而代之为一个叶节点,该叶节点所标识的类别由“大多数原则”确定。
               PEP算法是唯一使用Top-Down剪枝策略，这种策略会导致与先剪枝出现同样的问题，将该结点的某子节点不需要被剪枝时被剪掉；另外PEP方法会有剪枝失败的情况出现。
              虽然PEP方法存在一些局限性，但是在实际应用中表现出了较高的精度,。两外PEP方法不需要分离训练集合和验证机和，对于数据量比较少的情况比较有利。再者其剪枝策略比
              其它方法相比效率更高，速度更快。因为在剪枝过程中，树中的每颗子树最多需要访问一次，在最坏的情况下，它的计算时间复杂度也只和非剪枝树的非叶子节点数目成线性关系。
          
               Tmax:由决策树生成算法生成的未剪枝的完全决策树;
               Tt:以内部节点t为根的一棵子树;
               n(t):到达节点t的所有样本数目;
               ni(t):到达节点t且属于第i类的样本数目;
               e(t):到达节点t但不属于节点t所标识的类别的样本数目;
               r(t):错分样本率,其值为e(t)/n(t)。
               
                D                 【二维数组】 存放记录    
                L                 【一维数组】 存放类标志Ci（i=0..n）   
                penalty_term      【经验性的惩罚因子】 
            '''   
           
               # 自顶向下的顺序遍历完全决策树Tmax,对于每个内部节点t逐一比较, 如果剪枝剪掉了，就到下一个内部节点，否则继续。
           def recursion(T , D, L,penalty_term):
              assert len(D) == len(L)
              if(T.col == None) :
                  return              

                   # 决策树叶子节点个数
              def recursion1(T , leaf_count):
                  if(T.col == None) :
                      leaf_count[0] = leaf_count[0] + 1
                  for child in T.child:
                      recursion(child,leaf_count)
              leaf_count=[0]            
              recursion1(T,leaf_count)
                  # 罚项总和
              omega_T  = leaf_count[0]*penalty_term
                  #  未剪枝的错分样本数变为 e′(Tt)=∑[e(s)+1/2],s∈{Tt子树的所有叶节点};
              apply_error_rate , e_T_t = _apply_error(self.tree ,D, L)
              e_T_t = e_T_t + omega_T               
                  #  训练记录个数
              n_t = len(L)  
              
              # e'(Tt)的标准差，由于误差近似看成是二项式分布     
              # SE(e′(Tt)),SE(e′(Tt))=[e′(Tt)・(n(t)-e′(Tt))/n(t)]^1/2
              SE_e_T_t = sp.pow(e_T_t*(n_t-e_T_t)/float(n_t),0.5)

                   # 该叶节点所标识的类别的“大多数原则”
              def recursion2(T , majority_classc):
                  if(T.col != None) :
                      return
                  append (majority_classc,T.kind)
                  for child in T.child:
                      recursion(child,leaf_count)
              majority_classc=[]            
              recursion2(T,majority_classc)
              majority_classc = Counter(majority_classc)
              
                   #当满足条件e′(t)<=e′(Tt)+SE(e′(Tt))时,就进行剪枝,剪掉以t为根节点的子树Tt而代之为一个叶节点,该叶节点所标识的类别由“大多数原则”确定。
              if e_T_t <= 
                  
              for child in node.child:
                recursion(child)

           recursion(self.tree)
           pass 
               
     def MDL(self,D,L):      
        '''
              最小描述长度原则评估泛化误差 --- 该评估结合了模型复杂度（叶子节点的个数、中间节点个数，错误训练个数）
               D                 【二维数组】 存放记录    
               L                 【一维数组】 存放类标志Ci（i=0..n）   
       '''        
           # 决策书叶子节点个数和属性节点个数
        def recursion(node ,branch_count ,leaf_count):
            if(node.col == None) :
               leaf_count[0] = leaf_count[0] + 1
            else:
               branch_count[0] = branch_count[0] + 1
            for child in node.child:
               recursion(child,branch_count,leaf_count)
        leaf_count_k     =[0]
        branch_count_m   =[0]
        recursion(self.tree,branch_count_m,leaf_count_k)
         
           # 决策树所有节点的编码开销
        Cost_tree = sp.log2(branch_count_m[0])*branch_count_m[0]+sp.log2(leaf_count_k[0])*leaf_count_k[0]
           # 决策树总的训练误差
        apply_error_rate , e_T_= self.apply_error(D, L)
        
           #  训练记录个数
        N_t = len(L)
           
           # 在训练集上分类错误编码开销
        Cost_data_tree = sp.log2(N_t)*e_T_
        return    Cost_tree+Cost_data_tree
     
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
