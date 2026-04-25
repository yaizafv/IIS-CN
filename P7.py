import numpy as np
import matplotlib.pyplot as plt

def lagrange_fund(k, x, z):
    """
    Calcula el k-ésimo polinomio fundamental de Lagrange en los puntos z.
    """
    p = 1
    for i in range(len(x)):
        if i != k:
            p *= (z - x[i])/(x[k]-x[i])
    return p

# Nodos del problema
x = np.array([-1., 0, 2, 3, 5])
z = np.linspace(min(x), max(x), 100)

# Dibujar todos los polinomios fundamentales
plt.figure(figsize=(10, 6))
for k in range(len(x)):
    plt.plot(z, lagrange_fund(k, x, z), label=f'L{k}')
    plt.plot(x[k], 1, 'ro') # El polinomio vale 1 en su propio nodo
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.title("Polinomios Fundamentales de Lagrange")
plt.show()

#%%

def polinomio_lagrange(x, y, z):
    """
    Calcula el valor del polinomio interpolante de Lagrange en z.
    """
    pz = 0
    for k in range(len(x)):
        pz += y[k] * lagrange_fund(k, x, z)
    return pz

# Datos
y = np.array([1., 3, 4, 3, 1])

# Dibujar resultado
zp = np.linspace(min(x), max(x), 100)
plt.plot(zp, polinomio_lagrange(x, y, zp), label='P(x)')
plt.plot(x, y, 'ro', label='Puntos')
plt.legend()
plt.show()

#%%

import numpy.polynomial.polynomial as pol

def chebyshev_nodes(a, b, n):
    i = np.arange(1, n + 1)
    nodes_standard = np.cos((2*i - 1) * np.pi / (2*n))
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes_standard

def ejercicio3(f, a, b, n):
    x_eq = np.linspace(a, b, n)
    y_eq = f(x_eq)
    
    x_ch = chebyshev_nodes(a, b, n)
    y_ch = f(x_ch)
    
    xp = np.linspace(a, b, 200)
    
    p_eq = pol.polyfit(x_eq, y_eq, n-1)
    p_ch = pol.polyfit(x_ch, y_ch, n-1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xp, f(xp), 'k--', label='Original')
    plt.plot(xp, pol.polyval(xp, p_eq), 'r', label='Equiespaciados')
    plt.plot(x_eq, y_eq, 'ro')
    plt.axis([-1.05, 1.05, -0.3, 2.3])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(xp, f(xp), 'k--', label='Original')
    plt.plot(xp, pol.polyval(xp, p_ch), 'b', label='Chebyshev')
    plt.plot(x_ch, y_ch, 'bo')
    plt.axis([-1.05, 1.05, -0.3, 2.3])
    plt.legend()
    plt.show()

f1 = lambda x: 1 / (1 + 25*x**2)
ejercicio3(f1, -1, 1, 11)

#%%

def dif_div(x, y):
    n = len(y)
    coefs = np.zeros((n, n))
    coefs[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coefs[i,j] = (coefs[i+1,j-1] - coefs[i,j-1]) / (x[i+j] - x[i])
    return coefs

def polinomio_newton(x, y, z):
    coefs = dif_div(x, y)[0, :] # Primera fila contiene los coeficientes
    n = len(x)
    p = coefs[n-1]
    for k in range(1, n):
        p = coefs[n-1-k] + (z - x[n-1-k]) * p
    return p

# Comprobación con nodos del ejercicio
x_ex4 = np.array([-1., 0, 2, 3, 5])
y_ex4 = np.array([1., 3, 4, 3, 1])
print("Matriz de diferencias divididas:\n", dif_div(x_ex4, y_ex4))

#%%

from scipy.interpolate import interp1d, CubicSpline

x_ex5 = np.arange(11)
y_ex5 = np.cos(x_ex5)
xp = np.linspace(0, 10, 100)
f_real = np.cos(xp)

# Interpolaciones
f_lineal = interp1d(x_ex5, y_ex5, kind='linear')
f_spline = CubicSpline(x_ex5, y_ex5, bc_type='natural')

# Errores
err_lin = np.linalg.norm(f_lineal(xp) - f_real)
err_spl = np.linalg.norm(f_spline(xp) - f_real)

print(f"Error lineal a trozos: {err_lin:.5f}")
print(f"Error splines cúbicos: {err_spl:.5f}")

#%%

def Vandermonde(x):
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        V[:, i] = x**i
    return V

def polVandermonde(x, y):
    V = Vandermonde(x)
    p = np.linalg.solve(V, y)
    return p

# Ejecución
p_vander = polVandermonde(x_ex4, y_ex4)
print("Coeficientes (Vandermonde):", p_vander)