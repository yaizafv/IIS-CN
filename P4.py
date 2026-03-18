# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 18:08:24 2026

@author: UO301762
"""

import numpy as np

np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)

def triangulariza(A, b):
    n = len(b)
    At = np.copy(A)
    bt = np.copy(b)
    
    for i in range(n - 1):
        factor = At[i+1, i] / At[i, i]
        At[i+1, i] = 0
        At[i+1, i+1] = At[i+1, i+1] - factor * At[i, i+1]
        bt[i+1] = bt[i+1] - factor * bt[i]
    return At, bt

n1 = 7
A1_diag = np.diag(np.ones(n1)) * 3
A2_diag = np.diag(np.ones(n1-1), 1)
A_1 = A1_diag + A2_diag + A2_diag.T
b_1 = np.arange(n1, 2 * n1) * 1.

At1, bt1 = triangulariza(A_1, b_1)

print("SISTEMA 1 (n=7)")
print("Matriz At:\n", At1)
print("Vector bt:\n", bt1)
print("-" * 30)

n2 = 8
np.random.seed(3)
A1_rand = np.diag(np.random.rand(n2))
A2_rand = np.diag(np.random.rand(n2-1), 1)
A_2 = A1_rand + A2_rand + A2_rand.T
b_2 = np.random.rand(n2) 

At2, bt2 = triangulariza(A_2, b_2)

print("SISTEMA 2 (n=8, aleatorio)")
print("Matriz At:\n", At2)
print("Vector bt:\n", bt2)
    
# %%

def sust_reg(At, bt):
    n = len(bt)
    x = np.zeros(n)
    x[n-1] = bt[n-1] / At[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (bt[i] - At[i, i+1] * x[i+1]) / At[i, i]
    return x

n1 = 7
A1 = np.diag(np.ones(n1)) * 3
A2 = np.diag(np.ones(n1-1), 1)
A_s1 = A1 + A2 + A2.T
b_s1 = np.arange(n1, 2*n1) * 1.

At1, bt1 = triangulariza(A_s1, b_s1)
x1 = sust_reg(At1, bt1)

print("SOLUCIÓN SISTEMA 1")
print("x")
print(x1)
print("\n")

n2 = 8
np.random.seed(3)
A1_rand = np.diag(np.random.rand(n2))
A2_rand = np.diag(np.random.rand(n2-1), 1)
A_s2 = A1_rand + A2_rand + A2_rand.T
b_s2 = np.arange(n2, 2*n2) * 1. 

At2, bt2 = triangulariza(A_s2, b_s2)
x2 = sust_reg(At2, bt2)

print("SOLUCIÓN SISTEMA 2")
print("x")
print(x2)

# %% 

def multiplicar_3bucles(A, B):
    m, n = A.shape
    n_b, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]   
    return C

A = np.array([[-3., 2], [-2, 0], [-4, 4], [4, -4]])
B = np.array([[4., -3, 1], [-2, 1, 1]])

resultado1 = multiplicar_3bucles(A, B)
print("C1:\n", resultado1)

A1 = np.array([[-3., 2], [-2, 0], [-4, 4], [4, -4], [1, 1]])
B1 = np.array([[4., -3, 1, 1], [-2, 1, 1, 1]])

resultado2 = multiplicar_3bucles(A1, B1)
print("C2:\n", resultado2)

# %%

def multiplicar_2bucles(A, B):
    m = A.shape[0]
    p = B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C[i, j] = np.sum(A[i, :] * B[:, j])
    return C

A = np.array([[-16, 11, -1], [-8, 6, -2], [-24, 16, 0], [24, -16, 0]])
B = np.array([[-16, 11, -1, -1], [-8, 6, -2, -2], [-24, 16, 0, 0], [24, -16, 0, 0], [2, -2, 2, 2]])

resultado1 = multiplicar_2bucles(A, B)
print("C1:\n", resultado)

# %% 

for j in range(p):
    C[:,j] = np.sum(A*B[:,j],axis=1)
    
    
# %%


