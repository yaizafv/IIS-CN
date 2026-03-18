# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:20:02 2026

@author: uo301762
"""

import numpy as np
import matplotlib.pyplot as plt

# Ejercicio1

x0 = -0.4
tol = 1e-8
maxNumSum = 100

# Usamos recurrencia: term_{k+1} = term_k * x/(k+1)
aprox = 0.0
term = 1.0   # k=0: 1
k = 0
num_sumandos = 0

while (abs(term) > tol) and (num_sumandos < maxNumSum):
    aprox += term
    num_sumandos += 1
    k += 1
    term *= x0 / k

f = lambda x: np.exp(x)

print("Valor de la función en", x0, "=", f(x0))
print("Valor de la aproximación en", x0, "=", aprox)
print("Número de iteraciones =", num_sumandos)


#%%

# Ejercicio2

def funExp(x, tol, maxNumSum):
    y = np.zeros_like(x)
    termino = np.zeros_like(x)
    n = 1

    while (np.max(np.abs(termino)) > tol) and (n < maxNumSum):
        termino = termino * x / n
        y += termino
        n += 1

    return y

x = np.linspace(-1, 1, 50)
tol = 1.e-8
maxNumSum = 100

y_aprox = funExp(x, tol, maxNumSum)

f = lambda x: np.exp(x)
y = f(x)

plt.figure()
plt.plot(x,y,'y', linewidth = 4, label = 'f')
plt.plot(x,y,'b--', label = 'Aproximación f')
plt.title('Aproximación de f con el polinomio de McLaurin')
plt.show()


#%%

# Ejercicio3

x0 = np.pi / 4
tol = 1e-8
maxNumSum = 100

suma = 0.0
sumando = x0         
factorial = 1         
k = 0
numSum = 0
signo = 1

while abs(sumando) > tol and numSum < maxNumSum:
    suma += sumando
    numSum += 1
    k += 1
    signo *= -1
    factorial *= (2*k) * (2*k + 1)

    sumando = signo * x0**(2*k + 1) / factorial

f = lambda x: np.sin(x)
valor_exacto = f(x0)

print("Aproximación McLaurin :", suma)
print("Valor exacto          :", valor_exacto)
print("Error absoluto        :", abs(suma - valor_exacto))
print("Número de sumandos    :", numSum)



#%%

# Ejercicio 4

def funSin(x, tol=1e-8, maxNumSum=100):
    x = np.asarray(x, dtype=float)

    y_aprox = np.zeros_like(x)
    t = x.copy()  # t_0 = x
    i = 0

    while (np.max(np.abs(t)) > tol) and (i < maxNumSum):
        y_aprox += t
        # t_{i+1} = - t_i * x^2 / ((2i+2)(2i+3))
        denom = (2*i + 2) * (2*i + 3)
        t = -t * (x*x) / denom
        i += 1

    return y_aprox

x = np.linspace(-np.pi, np.pi, 50)
tol = 1e-8
maxNumSum = 100

y_aprox = funSin(x, tol, maxNumSum)
y_real = np.sin(x)

print("=== Ejercicio 4 ===")
plt.figure()
plt.plot(x, y_real, 'b--', label='sin(x) exacta')
plt.plot(x, y_aprox, 'y', linewidth=4, label='Aproximación McLaurin')
plt.legend()
plt.title("Aproximación del seno")
plt.grid(True, alpha=0.3)
plt.show()


#%%

# Ejercicio 5

x0 = 0.5
tol = 1e-4

# sinh: u_0 = x, u_{i+1} = u_i * x^2 / ((2i+2)(2i+3))   (todos positivos)
# cosh: v_0 = 1, v_{i+1} = v_i * x^2 / ((2i+1)(2i+2))
num = 0.0   # acumulado sinh
den = 0.0   # acumulado cosh

# iniciamos con los primeros términos explícitos
u = x0      # término i=0 de sinh
v = 1.0     # término i=0 de cosh
num += u
den += v

tanh_old = num / den
i = 0

while True:
    denom_u = (2*i + 2) * (2*i + 3)
    u = u * (x0*x0) / denom_u
    num += u

    denom_v = (2*i + 1) * (2*i + 2)
    v = v * (x0*x0) / denom_v
    den += v

    tanh_new = num / den

    if np.abs(tanh_new - tanh_old) < tol:
        break

    tanh_old = tanh_new
    i += 1

print("Valor aprox =", tanh_new)
print("Valor exacto =", np.tanh(x0))
print("Iteraciones (pares de términos añadidos) =", i+1)

# %% 

# Ejercicio 6
def funTanh(x, tol=1e-8):
    x = np.asarray(x, dtype=float)

    # Términos iniciales
    u = x.copy()                 # término i=0 de sinh
    v = np.ones_like(x)          # término i=0 de cosh
    num = u.copy()               # acumulado sinh
    den = v.copy()               # acumulado cosh

    tanh_old = num / den
    i = 0

    while True:
        # Siguiente término de sinh: u_{i+1} = u_i * x^2 / ((2i+2)(2i+3))
        denom_u = (2*i + 2) * (2*i + 3)
        u = u * (x*x) / denom_u
        num += u

        # Siguiente término de cosh: v_{i+1} = v_i * x^2 / ((2i+1)(2i+2))
        denom_v = (2*i + 1) * (2*i + 2)
        v = v * (x*x) / denom_v
        den += v

        tanh_new = num / den

        if np.max(np.abs(tanh_new - tanh_old)) < tol:
            break

        tanh_old = tanh_new
        i += 1

    return tanh_new

# --- Uso ---
x = np.linspace(-3, 3, 50)
tol = 1e-8

y_aprox = funTanh(x, tol)
y_real = np.tanh(x)

plt.figure()
plt.plot(x, y_real, 'b--', label='tanh exacta')
plt.plot(x, y_aprox, 'r', linewidth=3, label='tanh McLaurin')
plt.legend()
plt.title("Aproximación de tanh(x)")
plt.grid(True, alpha=0.3)
plt.show()