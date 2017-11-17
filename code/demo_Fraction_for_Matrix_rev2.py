# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:01:42 2017

@author: jack.zhou
"""


from fractions import Fraction



#首先把已给给定的矩阵里的元素全部化成Fraction对象
def change_to_frac(M):
    for i, r_e in enumerate(M):
        for j, l_e in enumerate(M[i]):
            M[i][j] = Fraction(M[i][j])
            
A = [['7', '5', '3', '-5'],
     ['-4', '6', '2', '-2'],
     ['-9', '4', '-5', '9'],
     ['-9', '-10', '5', '-4']]


#增广矩阵
def augmentMatrix(A, b):
    change_to_frac(A)
    change_to_frac(b)
    Ab = []
    if not len(A) == len(b):
        raise IndexError("Row number of A must be equal to row number of b")

    for i, ele in enumerate(A):
        Ab.append(ele + b[i])
    return Ab

#A = [['7', '5', '3', '-5'],
#     ['-4', '6', '2', '-2'],
#     ['-9', '4', '-5', '9'],
#     ['-9', '-10', '5', '-4']]
#
#b = [['1'],
#     ['1'],
#     ['1'],
#     ['1']]
#
#Ab = augmentMatrix(A, b)
#print(Ab)


###几种初等变换函数
# 第一种变换：对换 
#TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    change_to_frac(M)
    M[r1], M[r2] = M[r2], M[r1]
    
#第二种变换：某行乘以一个标量
# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    change_to_frac(M)
    if Fraction(scale) == Fraction(0):
        raise ValueError("Scale parameter must be non-zero value")
    for i,ele in enumerate(M[r]):
        M[r][i] = ele * Fraction(scale)
        
#第三种变换：r2行乘以一个标量加到r1行去        
# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    change_to_frac(M)
    if Fraction(scale) == Fraction(0):
        raise ValueError("Scale parameter must be non-zero")
    for i, ele in enumerate(M[r2]):
        M[r1][i] = M[r1][i] + ele * Fraction(scale)

#高斯消元函数
def gj_Solve(A, b):
    #求矩阵第col列中第i行到第j行的元素的绝对值的最大值,同时返回所在的行号,以及实际值
    def maxAbs_rowidx_InColumn(M,col,i,j):
        change_to_frac(M)
        max_val = Fraction(0) #max value will be found
        target_row_idx = i #index will be found
        act_val = M[i][col] #actual value will be found
        k = i
        while k <= j:
            val = abs(M[k][col])
            if val > max_val:
                max_val = val
                target_row_idx = k
                act_val = M[k][col]
            k += 1
        return target_row_idx, act_val, max_val
    
    
    change_to_frac(A)
    change_to_frac(b)
    ht_A = len(A)
    ht_b = len(b)
    if not ht_A == ht_b:
        return None
    Ab = augmentMatrix(A, b)
    for c, _ in enumerate(Ab):
        tg_row_idx, act_val, max_val = maxAbs_rowidx_InColumn(Ab, c, c, ht_A-1)
        if max_val == Fraction(0):
            return None
        if not tg_row_idx == c:
            #使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行 c）
            swapRows(Ab, c, tg_row_idx)
            
        #使用第二个行变换，将列c的对角线元素缩放为1
        scale = Fraction(1)/act_val
        scaleRow(Ab, c, scale)
        #多次使用第三个行变换，将列c的其他元素消为0
        #addScaledRow(M, r1, r2, scale)
        #TODO r1 <--- r1 + r2*scale
        tg_rows_set = set(t for t in range(ht_A)) - set({c})
        for r in tg_rows_set:
            scale = -Ab[r][c] #本来就是Fraction了
            if not abs(scale) == Fraction(0):
                addScaledRow(Ab, r, c, scale)
            
    result = []
    for t, r_list in enumerate(Ab):
        result.append([r_list[-1]])
    return result

#  -7,  -3,   1,  -9 ||  1  
#   0,   0,   0,   0 ||  1  
#  -2,   7,   7,  -3 ||  1  
#   8,  -5,  -6,   3 ||  1 
#-------------------------
#   7,   5,   3,  -5 ||  1  
#  -4,   6,   2,  -2 ||  1  
#  -9,   4,  -5,   9 ||  1  
#  -9, -10,   5,  -4 ||  1 


#A = [['7', '5', '3', '-5'],
#     ['-4', '6', '2', '-2'],
#     ['-9', '4', '-5', '9'],
#     ['-9', '-10', '5', '-4']]
#
#
#b = [['1'],
#     ['1'],
#     ['1'],
#     ['1']]
#
#ret = gj_Solve(A, b)
#print(ret)


#打印消元的关键步骤的结果
def print_Solve(A, b):
    #求矩阵第col列中第i行到第j行的元素的绝对值的最大值,同时返回所在的行号,以及实际值
    def maxAbs_rowidx_InColumn(M,col,i,j):
        change_to_frac(M)
        max_val = Fraction(0) #max value will be found
        target_row_idx = i #index will be found
        act_val = M[i][col] #actual value will be found
        k = i
        while k <= j:
            val = abs(M[k][col])
            if val > max_val:
                max_val = val
                target_row_idx = k
                act_val = M[k][col]
            k += 1
        return target_row_idx, act_val, max_val
    
    
    change_to_frac(A)
    change_to_frac(b)
    ht_A = len(A)
    ht_b = len(b)
    if not ht_A == ht_b:
        return None
    Ab = augmentMatrix(A, b)
    for c, _ in enumerate(Ab):
        tg_row_idx, act_val, max_val = maxAbs_rowidx_InColumn(Ab, c, c, ht_A-1)
        if max_val == Fraction(0):
            return None
        if not tg_row_idx == c:
            #使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行 c）
            swapRows(Ab, c, tg_row_idx)
            
        #使用第二个行变换，将列c的对角线元素缩放为1
        scale = Fraction(1)/act_val
        scaleRow(Ab, c, scale)
        #多次使用第三个行变换，将列c的其他元素消为0
        #addScaledRow(M, r1, r2, scale)
        #TODO r1 <--- r1 + r2*scale
        tg_rows_set = set(t for t in range(ht_A)) - set({c})
                    
        for r in tg_rows_set:
            scale = -Ab[r][c] #本来就是Fraction了
            if not abs(scale) == Fraction(0):
                addScaledRow(Ab, r, c, scale)
                
                
        show_matrix = []
        for rw, _ in enumerate(Ab):
            temp = []
            for cl, _ in enumerate(Ab[rw]):
                if Ab[rw][cl].denominator == 1:
                    ss = "{}".format(Ab[rw][cl].numerator)                  
                else:
                    ss = "{}/{}".format(Ab[rw][cl].numerator, Ab[rw][cl].denominator)                   
                temp.append(ss)
            show_matrix.append(temp)
        print("第{}个循环的结果是：\n{}".format(c + 1, show_matrix))
            


#A = [['7', '5', '3', '-5'],
#     ['-4', '6', '2', '-2'],
#     ['-9', '4', '-5', '9'],
#     ['-9', '-10', '5', '-4']]
#
#
#b = [['1'],
#     ['1'],
#     ['1'],
#     ['1']]

#A = [[7, 5, 3, -5],
#     [-4, 6, 2, -2],
#     [-9, 4, -5, 9],
#     [-9, -10, 5, -4]]
#
#
#b = [[1],
#     [1],
#     [1],
#     [1]]
#
#print_Solve(A, b)

#  -7,  -3,   1,  -9 ||  1  
#   0,   0,   0,   0 ||  1  
#  -2,   7,   7,  -3 ||  1  
#   8,  -5,  -6,   3 ||  1 
A = [[-7, -3, 1, -9],
     [0, 0, 0, 0],
     [-2, 7, 7, -3],
     [8, -5, -6, 3]]
   
b = [[1],
     [1],
     [1],
     [1]]

print_Solve(A, b)


