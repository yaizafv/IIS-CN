# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 18:07:57 2026

@author: uo301762
"""

import numpy as np

def busquedaIncremental(f, a, b, n):
    dx = (b - a) / n
    
    intervalos = np.zeros((n, 2))
    contador = 0
    
    x = np.linspace(a, b, n+1)

    for k in range(1, n+1):
        x_left = x[k-1]
        x_right = x[k]
        
        if f(x_left) * f(x_right) < 0:
            intervalos[contador, :] = [x_left, x_right]
            contador += 1

    return intervalos[:contador, :]

f1 = lambda x: x**5 - 3*x**2 + 1.6

intervalos_f1 = busquedaIncremental(f1, -1, 1.5, 25)
print("Intervalos que contienen raíces de f1:")
print(intervalos_f1)

f2 = lambda x: (x + 2) * np.cos(2*x)

intervalos_f2 = busquedaIncremental(f2, 0, 10, 100)
print("Intervalos que contienen raíces de f2:")
print(intervalos_f2)

#%%

def biseccion(f, a, b, tol=1e-6, maxiter=100):
    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0 
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos (Teorema de Bolzano).")
    for k in range(1, maxiter + 1):
        x = (a + b) / 2.0
        fx = f(x)
        if fx == 0:
            return x, k
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
        if (b - a) / 2.0 < tol:
            return (a + b) / 2.0, k
    return (a + b) / 2.0, maxiter

f1 = lambda x: x**5 - 3*x**2 + 1.6

def busquedaIncremental(f, a, b, n):
    dx = (b - a) / n
    intervalos = np.zeros((n, 2))
    contador = 0
    x = np.linspace(a, b, n+1)
    for k in range(1, n+1):
        xl, xr = x[k-1], x[k]
        if f(xl) * f(xr) < 0:
            intervalos[contador, :] = [xl, xr]
            contador += 1
    return intervalos[:contador, :]

intervalos_f1 = busquedaIncremental(f1, -1, 1.5, 25)

tol = 1e-6
MaxIter = 100

r_f1 = np.zeros(3)
iters_f1 = np.zeros(3, dtype=int)
for i in range(min(3, len(intervalos_f1))):
    a_i, b_i = intervalos_f1[i]
    r, k = biseccion(f1, a_i, b_i, tol=tol, maxiter=MaxIter)
    r_f1[i] = r
    iters_f1[i] = k

np.set_printoptions(precision=5, suppress=True)
print("Raíces aproximadas de f1:", r_f1)
print("Iteraciones por raíz:    ", iters_f1)


f2 = lambda x: np.cos(x) - 0.2*x**3 + 1.0/(x**2 + 1)

intervalos_f2 = busquedaIncremental(f2, -3, 3, 200)

raices_f2 = []
iters_f2 = []
for a_i, b_i in intervalos_f2:
    r, k = biseccion(f2, a_i, b_i, tol=tol, maxiter=MaxIter)
    raices_f2.append(r)
    iters_f2.append(k)

raices_f2 = np.array(raices_f2, dtype=float)
iters_f2  = np.array(iters_f2,  dtype=int)

np.set_printoptions(precision=5, suppress=True)
print("Raíces aproximadas de f2:", raices_f2)
print("Iteraciones por raíz:    ", iters_f2)

import matplotlib.pyplot as plt

def plot_func_y_raices(f, a, b, raices, titulo=''):
    X = np.linspace(a, b, 600)
    plt.figure()
    plt.plot(X, f(X), label='f(x)')
    plt.plot(X, 0*X, 'k-')
    raices = np.array(raices, dtype=float)
    if raices.size > 0:
        plt.plot(raices, np.zeros_like(raices), 'ro', label='raíces (aprox.)')
    plt.legend()
    plt.title(titulo)
    plt.show()

plot_func_y_raices(f1, -1, 1.5, r_f1, titulo='f1 y raíces por bisección')
plot_func_y_raices(f2, -3, 3, raices_f2, titulo='f2 y raíces por bisección')

#%%

def newton(f, df, x0, tol=1e-6, maxiter=100):
    x_prev = x0

    for k in range(1, maxiter + 1):
        fx = f(x_prev)
        dfx = df(x_prev)
        if dfx == 0:
            raise ZeroDivisionError("Derivada nula. Newton no puede continuar.")
        x = x_prev - fx/dfx
        if abs(x - x_prev) < tol:
            return x, k
        x_prev = x
    return x_prev, maxiter

f1 = lambda x: x**5 - 3*x**2 + 1.6
df1 = lambda x: 5*x**4 - 6*x 

intervalos_f1 = busquedaIncremental(f1, -1, 1.5, 25)

tol = 1e-6
MaxIter = 100

raices_f1 = np.zeros(3)
iters_f1  = np.zeros(3, dtype=int)

for i in range(3):
    x0 = intervalos_f1[i, 0]   
    r, k = newton(f1, df1, x0, tol, MaxIter)
    raices_f1[i] = r
    iters_f1[i]  = k

np.set_printoptions(precision=6, suppress=True)
print("Raíces (Newton) f1:", raices_f1)
print("Iteraciones:", iters_f1)

f2  = lambda x: np.cos(x) - 0.2*x**3 + 1/(x**2 + 1)

df2 = lambda x: -np.sin(x) - 0.6*x**2 - (2*x)/(x**2 + 1)**2

x0_list = [-2, -0.8, 1.2]

raices_f2 = np.zeros(len(x0_list))
iters_f2  = np.zeros(len(x0_list), dtype=int)

tol = 1e-6
MaxIter = 100

for i, x0 in enumerate(x0_list):
    r, k = newton(f2, df2, x0, tol, MaxIter)
    raices_f2[i] = r
    iters_f2[i]  = k

print("Raíces (Newton) f2:", raices_f2)
print("Iteraciones:", iters_f2)

import matplotlib.pyplot as plt

def plot_func_y_raices(f, a, b, raices, titulo=''):
    X = np.linspace(a, b, 600)
    plt.figure()
    plt.plot(X, f(X), label='f(x)')
    plt.axhline(0, color='k')
    raices = np.array(raices)
    plt.plot(raices, np.zeros_like(raices), 'ro', label='raíces')
    plt.title(titulo)
    plt.legend()
    plt.show()

plot_func_y_raices(f1, -1, 1.5, raices_f1, "Newton — f1")
plot_func_y_raices(f2, -3, 3, raices_f2, "Newton — f2")
