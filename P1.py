import numpy as np
import matplotlib.pyplot as plt

# Ejercicio 1. Crea e imprime los siguientes vectores y matrices numpy

a = np.array([1, 3, 7])

b = np.array([[2, 4, 3],
              [0, 1, 6]])

c = np.ones(3)
d = np.zeros(4)

e = np.zeros((3, 2))
f = np.ones((3, 4))

print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)
print("f =", f)

#%%

# Ejercicio 2. Utilizando primero la orden arange y luego linspace crea los vectores:
# a = (7, 9, 11, 13, 15)
# b = (10, 9, 8, 7, 6)
# c = (15, 10, 5, 0)
# d es un vector que empieza en 0 y acaba en 1 y contiene 11 puntos equiespaciados.
# e es un vector que empieza en -1 y acaba en 1 y sus puntos divididen [−1, 1] en 10 intervalos iguales.
# f es un vector que empieza en 1 y acaba en 2 y sus puntos están separados por un paso 0.1

np.set_printoptions(precision=2, suppress=True)     # para que el formato no sea exponencial y use dos dígitos.

# arange
a = np.arange(7, 16, 2)
b = np.arange(10, 5, -1)
c = np.arange(15, -1, -5)
f = np.arange(1, 2.1, 0.1)

# linspace
d = np.linspace(0, 1, 11)
e = np.linspace(-1, 1, 11)

print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)
print("f =", f)

#%%

# Ejercicio 3. Utilizando arange, crea el vector:
# v =(0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0, 12.1) 

v = np.arange(0, 13.2, 1.1)

vi = v[::-1]

v1 = v[::2]
v2 = v[1::2]

v1_3 = v[::3]
v2_3 = v[1::3]
v3_3 = v[2::3]

v1_4 = v[::4]
v2_4 = v[1::4]
v3_4 = v[2::4]
v4_4 = v[3::4]

print("v =", v)
print("vi =", vi)

print("v1 =", v1)
print("v2 =", v2)

print("v1 =", v1_3)
print("v2 =", v2_3)
print("v3 =", v3_3)

print("v1 =", v1_4)
print("v2 =", v2_4)
print("v3 =", v3_4)
print("v4 =", v4_4)

#%%

# Ejercicio 4. A partir de un vector a = (1, 2, 3) crear un vector b = (0, 1, 2, 3, 0).
# Usando np.append dos veces: añadir un cero al final y luego gira el vector añadir el otro cero y volver a girar el vector.
# Creando un vector b = (0, 0, 0, 0, 0) e insertando el vector a con slicing.
# Usando np.concatenate creando un vector c = (0) previamente.
a = np.array([1, 2, 3])
b = np.append(a, 0)
b = np.append(0, b)
print("b =", b)

#%%

# Ejercicio 5
A = np.array([[ 2,  1,  3,  4], [ 9,  8,  5,  7], [ 6, -1, -2, -8], 
              [-5, -7, -9, -6]])

a = A[:, 0]
b = A[2]
c = A[:2, :2]
d = A[2:, 2:]
e = A[1:3, 1:3]
f = A[:, 1:]
g = A[1:, 1:3]

print("A =\n", A)
print("a =", a)
print("b =", b)
print("c =\n", c)
print("d =\n", d)
print("e =\n", e)
print("f =\n", f)
print("g =\n", g)

#%%

# Ejercicio 6
f1 = lambda x: x*np.exp(x)
print(f1(2))

f2 = lambda z: z / (np.sin(z) * np.cos(z))
print(f2(np.pi / 4))

f3  = lambda x, y: (x*y)/(x**2 + y**2)
print(f3(2,4))

#%%

# Ejercicio 7. Dibuja en el intervalo [−2π, 2π] la función f(x) = xsen(3x)
x = np.linspace(2 * np.pi, -2 * np.pi, 200)
f = lambda x: x * np.sin(3*x)
OX = 0 * x
plt.figure()
plt.plot(x,f(x))                   # dibujar la función
plt.plot(x,OX,'k-')                # dibujar el eje X
plt.xlabel('x')
plt.ylabel('y')
plt.title('función')
plt.show()














