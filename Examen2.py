import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
# Cuatro decimales para arrays numpy y con ceros finales
np.set_printoptions(precision=4,floatmode='fixed',suppress=True)
#
NIg = 15 # Número de Iguales (=) para el título de cada ejercicio.

#%% Ejercicio 1
def dif_div(x,y):
    n = len(y)
    coefs = np.zeros((n, n))
    coefs[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coefs[i,j] = (coefs[i+1,j-1] - coefs[i,j-1]) / (x[i+j] - x[i])
    return coefs
#
def polinomio_newton(x,y,z):
    tabla = dif_div(x, y)
    coef = tabla[0, :] 
    n = len(coef)
    resultado = coef[n-1]
    for i in range(n-2, -1, -1):
        resultado = resultado * (z - x[i]) + coef[i]
    return resultado
#
def ejer1():
  print(f'{NIg*"="} {os.path.basename(__file__)}: {sys._getframe().f_code.co_name} {NIg*"="}')
  def ejem1():
    print('# Ejemplo 1:')
    np.random.seed(7)
    x = np.array(sorted(np.random.randint(20,size=(5))))
    y = np.random.randint(20,size=(5))
    print(f'x = {x}')
    print(f'y = {y}')
    tabla = dif_div(x, y)
    print('D =')
    print(tabla)
    print(f'c = {tabla[0,:]}')
    
    zp = np.linspace(min(x), max(x), 100)
    plt.plot(zp, polinomio_newton(x, y, zp), label='Polinomio')
    plt.plot(x, y, 'ro', label='Puntos')
    plt.legend()
    plt.title("Polinomio de interpolacion en la forma de Newton")
    plt.show()

    
  ejem1()
  def ejem2():
    print('# Ejemplo 2:')
    np.random.seed(33)
    x = np.array(sorted(np.random.randint(20,size=(8))))
    y = np.random.randint(20,size=(8))
    print(f'x = {x}')
    print(f'y = {y}')
    tabla = dif_div(x, y)
    print('D =')
    print(tabla)
    print(f'c = {tabla[0,:]}')
    
    plt.figure()
    zp = np.linspace(min(x), max(x), 100)
    plt.plot(zp, polinomio_newton(x, y, zp), label='Polinomio')
    plt.plot(x, y, 'ro', label='Puntos')
    plt.legend()
    plt.title("Polinomio de interpolacion en la forma de Newton")
    plt.show()
    
  ejem2()

#%% Ejercicio 2
def trapecio_compuesta(f,a,b,n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    Ia = h * (0.5*f(a) + np.sum(f(x[1:-1])) + 0.5*f(b))
    return Ia

def simpson_compuesta(f,a,b,n):
    h = (b - a) / n
    suma = 0
    for i in range(1, n+1):
        xi = a + i*h
        xim = xi - h/2
        suma += f(xi-h) + 4*f(xim) + f(xi)
    Ia = h * suma / 6
    return Ia

def ejer2():
  print(f'{NIg*"="} {os.path.basename(__file__)}: {sys._getframe().f_code.co_name} {NIg*"="}')
  def ejem1():
    print('# Ejemplo 1:')
    array = np.linspace(2, 9, 15)
    print(f' x = {array}')
    a = 2; b = 9; n = 15
    f = lambda x: 3 + np.sin(x)**2*np.cos(x)**2
    I_trap = trapecio_compuesta(f, a, b, n)
    I_exact, _ = quad(f, a, b)
    err_trap = abs(I_exact - I_trap)

    print(f'Ie = {I_exact: }')
    print(f'Ia = {I_trap: }')
    print(f'Ea = {err_trap: }')
    
  ejem1()
  def ejem2():
    print('# Ejemplo 2:')
    array = np.linspace(2, 9, 15)
    print(f' x = {array}')
    a = 2; b = 9; n = 15
    g = lambda x: 3 + np.sin(x)**2*np.cos(x)**2
    I_simp = simpson_compuesta(g, a, b, n)
    I_exact, _ = quad(g, a, b)
    err_simp = abs(I_exact - I_simp)
    print(f'Ie = {I_exact: }')
    print(f'Ia = {I_simp: }')
    print(f'Ea = {err_simp: }')
  ejem2()


def main():
  ejer1()
  ejer2()

if __name__ == "__main__":
  main()
#%%  
