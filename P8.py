# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:23:19 2026

@author: yaiza
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def dibujo(f, a, b, nodos):
    x_curva = np.linspace(a, b, 200)
    y_f = f(x_curva)
    
    y_nodos = f(nodos)
    grado = len(nodos) - 1
    coefs = np.polyfit(nodos, y_nodos, grado)
    p = np.poly1d(coefs)
    y_p = p(x_curva)
    
    plt.figure(figsize=(10, 6))
    
    plt.fill_between(x_curva, y_f, color='blue', alpha=0.4, label='Área Exacta')
    plt.plot(x_curva, y_f, 'b-', label='$f(x)$')
    
    plt.fill_between(x_curva, y_p, color='red', alpha=0.3, label='Área Aproximada (Polinomio)')
    plt.plot(x_curva, y_p, 'r--', label='Polinomio Interpolador')
    
    plt.plot(nodos, y_nodos, 'ko', label='Nodos')
    
    plt.title(f'Integración Numérica (n={len(nodos)} nodos)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

f1 = lambda x: np.exp(x)
a1, b1 = 0, 3
nodos1 = np.array([1, 2, 2.5])
dibujo(f1, a1, b1, nodos1)

f2 = lambda x: np.cos(x) + 1.5
a2, b2 = -3, 3
nodos2 = np.array([-3., -1, 0, 1, 3])
dibujo(f2, a2, b2, nodos2)

x = sym.Symbol('x', real=True)
I_exacta = sym.integrate(sym.log(x), (x, 1, 3))
I_exacta = float(I_exacta)
print(I_exacta)

def trapecio(f, a, b):
    return (b - a) * (f(a) + f(b)) / 2

f = lambda x: np.log(x)

print("Aprox:", trapecio(I_exacta,1,3))
print("Exacta:", I_exacta)

def trapecio(f, a, b):
    return (b - a) * (f(a) + f(b)) / 2

print("Aprox:", trapecio(f,1,3))
print("Exacta:", I_exacta)

def simpson(f, a, b):
    return (b - a) * (f(a) + 4*f((a+b)/2) + f(b)) / 6

print("Aprox:", simpson(f,1,3))
print("Exacta:", I_exacta)

def punto_medio_comp(f, a, b, n):
    h = (b - a) / n
    x_medios = a + h*(np.arange(n) + 0.5)
    return h * np.sum(f(x_medios))

print("Aprox:", punto_medio_comp(f,1,3,5))
print("Exacta:", I_exacta)

def trapecio_comp(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (0.5*f(a) + np.sum(f(x[1:-1])) + 0.5*f(b))

print("Aprox:", trapecio_comp(f,1,3,4))
print("Exacta:", I_exacta)

def simpson_comp(f, a, b, n):
    h = (b - a) / n
    suma = 0
    for i in range(1, n+1):
        xi = a + i*h
        xim = xi - h/2
        suma += f(xi-h) + 4*f(xim) + f(xi)
    return h * suma / 6

print("Aprox:", simpson_comp(f,1,3,4))
print("Exacta:", I_exacta)

#%%
    





