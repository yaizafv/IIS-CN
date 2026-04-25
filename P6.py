# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:12:54 2026

@author: yaiza
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
from scipy.integrate import quad
from scipy.special import eval_legendre
np.set_printoptions(precision=2, suppress=True)

def aprox1(f, g, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    V = np.ones((n, 1))
    X = np.array([x]).T
    for i in range(1, g + 1):
        V = np.concatenate((V, (X**i)), axis = 1)
    C = np.dot(V.T, V)
    d = np.dot(V.T, y)
    p = np.linalg.solve(C, d)
    
    x_plot = np.linspace(a, b, 50)
    y_plot = pol.polyval(x_plot, p)
    plt.scatter(x, y, color='red', label='puntos')
    plt.plot(x_plot, y_plot, label='función aproximada')
    plt.legend()
    plt.show()
    return p

f1 = lambda x: np.sin(x)
aprox1(f1, 2, 0, 2, 5)

#%%

def aprox2(f, g, a, b):
    # 1. Construir matriz C y vector d mediante integrales
    C = np.zeros((g + 1, g + 1))
    d = np.zeros(g + 1)
    
    for i in range(g + 1):
        for j in range(g + 1):
            C[i, j] = quad(lambda x: x**(i + j), a, b)[0]
        d[i] = quad(lambda x: (x**i) * f(x), a, b)[0]
    
    # 2. Resolver
    p = np.linalg.solve(C, d)
    
    # 3. Dibujar
    x_plot = np.linspace(a, b, 100)
    plt.plot(x_plot, f(x_plot), 'r', label='función exacta')
    plt.plot(x_plot, pol.polyval(x_plot, p), 'k--', label='función aproximada')
    plt.legend()
    plt.show()
    return p

#%%

def aprox3(f, g, a, b):

    # Transformación al intervalo [-1,1]
    T = lambda x: (2*x - (a+b))/(b-a)

    # Coeficientes
    a_k = []
    for k in range(g+1):
        num = quad(lambda x: f(x)*eval_legendre(k, T(x)), a, b)[0]
        den = quad(lambda x: eval_legendre(k, T(x))**2, a, b)[0]
        a_k.append(num/den)

    # Evaluación del polinomio
    xx = np.linspace(a, b, 200)
    yy = np.zeros_like(xx)
    for k in range(g+1):
        yy += a_k[k] * eval_legendre(k, T(xx))

    # Dibujar
    plt.figure(figsize=(7,4))
    plt.plot(xx, f(xx), label='Función')
    plt.plot(xx, yy, label='Aproximación Legendre')
    plt.legend()
    plt.show()


# Aproximación grado 4 de cos(x) en [0,2]
aprox3(lambda x: np.cos(x), 4, 0, 2)

# Aproximación grado 4 de f2
aprox3(lambda x: np.cos(np.arctan(x)) - np.log(x+5), 4, -2, 0)

#%%

def fourier(f, n, T, a0_only=False):
    a0 = (1/T) * quad(lambda x: f(x), 0, T)[0]
    a = [a0]

    if a0_only:
        return a, [0], [0]

    ak = []
    bk = []

    for k in range(1, n+1):
        ak.append((2/T) * quad(lambda x: f(x)*np.cos(2*np.pi*k*x/T), 0, T)[0])
        bk.append((2/T) * quad(lambda x: f(x)*np.sin(2*np.pi*k*x/T), 0, T)[0])

    return a, ak, bk


def plot_fourier(f, n, T):
    a0, ak, bk = fourier(f, n, T)

    xx = np.linspace(0, T, 1000)
    yy = np.zeros_like(xx) + a0[0]/2

    for k in range(1, n+1):
        yy += ak[k-1]*np.cos(2*np.pi*k*xx/T)
        yy += bk[k-1]*np.sin(2*np.pi*k*xx/T)

    plt.figure(figsize=(7,4))
    plt.plot(xx, f(xx), label='Función')
    plt.plot(xx, yy, label=f'Serie Fourier orden {n}')
    plt.legend()
    plt.show()


# Ejercicio 5:
# f(x)=x en [0,3]
plot_fourier(lambda x: x, 5, 3)

# f(x)=(x-π)^2 en [0,2π]
plot_fourier(lambda x: (x-np.pi)**2, 6, 2*np.pi)

