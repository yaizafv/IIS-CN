# -*- coding: utf-8 -*-
"""
Archivo único con la solución de los Ejercicios 1–7 del documento
"Introducción a Python para computación numérica II".

Convenciones del documento:
- Los polinomios se representan con coeficientes en **orden ascendente**
  (formato `numpy.polynomial.polynomial`):
    p = [p0, p1, ..., p_{n-1}]  =>  P(x) = p0 + p1*x + ... + p_{n-1}*x^{n-1}
- Para evaluar P(x) de forma nativa se puede usar: `pol.polyval(x, p)`

Contenido:
- Ejercicio 1:  horner(x0, p)  => (cociente, resto)
- Ejercicio 2:  HornerV(x, p)  => y (evaluación punto a punto con bucle externo sobre x)
- Ejercicio 3:  dersuc_punto(x0, p) y demostración de remainders/derivadas
- Ejercicio 4:  divisores(m) y raices_simples(p)
- Ejercicio 5:  raices_multiples(p)
- Ejercicio 6:  hornerVect(x, p) (vectorizado, sin bucle sobre x) + comparación de tiempos
- Ejercicio 7:  derivadasSuc(x, p) (matriz con P, P', P'', ..., P^{(n-1)}) y gráficos

El código está pensado para ejecutarse como script. Requiere numpy y matplotlib.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

# ============================
# Ejercicio 1: horner(x0, p)
# ============================

def horner(x0: float, p: np.ndarray):
    """Divide el polinomio P por (x - x0) usando Horner (Ruffini) y devuelve
    (cociente, resto).

    Parámetros
    ----------
    x0 : float
        Punto en el que dividimos por (x - x0).
    p : np.ndarray (1D)
        Coeficientes en orden ascendente (formato pol).

    Devuelve
    --------
    cociente : np.ndarray
        Coeficientes del polinomio cociente, en orden ascendente.
    resto : float
        Resto de la división, que coincide con P(x0).

    Nota: Implementación adaptada a coeficientes en orden ascendente.
    - Recorremos de mayor a menor grado manteniendo una variable acumuladora.
    - En cada paso guardamos en q[k] el acumulado ANTES de incorporar p[k].
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        raise ValueError("El vector de coeficientes no puede estar vacío.")
    if n == 1:
        # P(x) = p0; al dividir por (x-x0) el cociente es 0 y resto=p0
        return np.zeros(0, dtype=float), float(p[0])

    q = np.zeros(n - 1, dtype=float)
    y = p[-1]  # acumulador inicial (coeficiente del mayor grado)
    # Recorremos del penúltimo coeficiente al primero
    for k in range(n - 2, -1, -1):
        q[k] = y
        y = p[k] + x0 * y
    resto = float(y)
    return q, resto


# ==========================================
# Ejercicio 2: HornerV(x, p) (bucle sobre x)
# ==========================================

def HornerV(x: np.ndarray, p: np.ndarray):
    """Evalúa P en todos los puntos del vector x, repitiendo el proceso de
    Horner para cada x[i]. Devuelve y con la misma forma que x.

    Se implementa con un bucle externo sobre x, como pide el enunciado.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    for i, x0 in enumerate(x):
        # Solo nos interesa el resto (P(x0))
        _, r = horner(x0, p)
        y[i] = r
    return y


# ==================================================
# Ejercicio 6 (parte 1): hornerVect(x, p) vectorizado
# ==================================================

def hornerVect(x: np.ndarray, p: np.ndarray):
    """Evalúa P en todos los puntos de x con vectorización (sin bucle sobre x).
    Mantiene un pequeño bucle sobre los coeficientes (necesario y habitual
    en un Horner vectorizado).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return np.zeros_like(x, dtype=float)
    y = np.full_like(x, p[-1], dtype=float)
    for k in range(len(p) - 2, -1, -1):
        y = p[k] + x * y
    return y


# ===============================================
# Ejercicio 3: derivadas sucesivas en un solo punto
# ===============================================

def dersuc_punto(x0: float, p: np.ndarray):
    """Calcula, mediante aplicaciones sucesivas de Horner, los restos y las
    derivadas sucesivas de P en x0.

    Devuelve
    --------
    restos : np.ndarray
        Vector r[i] = resto al dividir sucesivamente por (x - x0).
    derivadas : np.ndarray
        Vector d[i] = P^{(i)}(x0) para i=0..n-1. Se obtiene como r[i] * i!
        (con factorial construido iterativamente, sin usar funciones de factorial).
    """
    p_work = np.array(p, dtype=float)
    n = len(p_work)
    restos = np.zeros(n, dtype=float)
    derivadas = np.zeros(n, dtype=float)

    fact = 1.0  # 0! = 1
    for i in range(n):
        # resto de dividir p_work entre (x - x0)
        q, r = horner(x0, p_work)
        restos[i] = r
        derivadas[i] = r * fact
        # preparar para la siguiente derivada (siguiente división)
        p_work = q
        if i + 1 < n:
            fact *= (i + 1)  # (i+1)!
    return restos, derivadas


# ===============================================
# Ejercicio 7: derivadas sucesivas para vector x
# ===============================================

def derivadasSuc(x: np.ndarray, p: np.ndarray):
    """Para cada x[i] calcula [P(x[i]), P'(x[i]), ..., P^{(n-1)}(x[i])].

    Devuelve una matriz Y de tamaño (m, n), donde m = len(x) y n = len(p).
    Se apoya en `dersuc_punto` para cada punto x[i].
    """
    x = np.asarray(x, dtype=float)
    m = len(x)
    n = len(p)
    Y = np.zeros((m, n), dtype=float)
    for i in range(m):
        _, d = dersuc_punto(x[i], p)
        Y[i, :] = d
    return Y


# ==================================
# Ejercicio 4: divisores y raíces Z
# ==================================

def divisores(m: int):
    """Devuelve un array numpy con los divisores enteros positivos de m y sus
    opuestos, en el orden: 1, -1, 2, -2, 3, -3, ...
    """
    m = abs(int(m))
    if m == 0:
        return np.array([0.0])
    # En el peor caso hay 2*m posiciones (1..m y sus opuestos); recortamos luego
    div = np.zeros(2 * m, dtype=float)
    idx = 0
    for d in range(1, m + 1):
        if m % d == 0:
            div[idx] = float(d); idx += 1
            div[idx] = float(-d); idx += 1
    return div[:idx]


def raices_simples(p: np.ndarray, atol: float = 1e-12):
    """Calcula raíces ENTERAS SIMPLES de un polinomio mónico con coeficientes en
    orden ascendente. Devuelve un array con las raíces (cada una una sola vez).

    Estrategia: probamos sucesivamente divisores del término independiente
    (p[0]). Al encontrar raíz exacta (resto ~ 0), la almacenamos y continuamos
    con el cociente. Se asume que todas las raíces son enteras y simples.
    """
    p_work = np.array(p, dtype=float)
    if not np.isclose(p_work[-1], 1.0):
        # Si no es mónico, normalizamos para usar el criterio de divisores de p[0]
        p_work = p_work / p_work[-1]
    roots = []
    while len(p_work) > 1:
        cand = divisores(p_work[0])  # divisores del término independiente
        found = False
        for x0 in cand:
            q, r = horner(x0, p_work)
            if abs(r) <= atol:
                roots.append(float(x0))
                p_work = q
                found = True
                break  # volvemos a probar desde el primer divisor con el nuevo polinomio
        if not found:
            # No se encontró raíz entera (según hipótesis no debería ocurrir)
            break
    return np.array(roots, dtype=float)


def raices_multiples(p: np.ndarray, atol: float = 1e-12):
    """Calcula raíces ENTERAS (permitiendo multiplicidades). Cada raíz aparece
    tantas veces como su multiplicidad.
    """
    p_work = np.array(p, dtype=float)
    if not np.isclose(p_work[-1], 1.0):
        p_work = p_work / p_work[-1]
    roots = []
    while len(p_work) > 1:
        cand = divisores(p_work[0])
        progress = False
        for x0 in cand:
            q, r = horner(x0, p_work)
            if abs(r) <= atol:
                # Encontrada raíz; seguir dividiendo por el mismo x0 hasta que deje de serlo
                while abs(r) <= atol and len(p_work) > 1:
                    roots.append(float(x0))
                    p_work = q
                    if len(p_work) > 1:
                        q, r = horner(x0, p_work)
                progress = True
                break
        if not progress:
            break
    return np.array(roots, dtype=float)


# =============================
# main(): pruebas y visualización
# =============================

def main():
    np.set_printoptions(suppress=True)

    # ------------------
    # Ejercicio 1 (pruebas)
    # ------------------
    print("Ejercicio 1")
    p0 = np.array([1., 2, 1])
    x0 = 1.0
    c0, r0 = horner(x0, p0)
    rp0 = pol.polyval(x0, p0)
    print("Coeficientes de Q =", c0)
    print("P0(1) =", r0)
    print("Con polyval =", rp0)

    p1 = np.array([1., -1, 2, -3, 5, -2])
    x1 = 1.0
    c1, r1 = horner(x1, p1)
    rp1 = pol.polyval(x1, p1)
    print("Coeficientes de Q =", c1)
    print("P1(1) =", r1)
    print("Con polyval =", rp1)

    p2 = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    x2 = -1.0
    c2, r2 = horner(x2, p2)
    rp2 = pol.polyval(x2, p2)
    print("Coeficientes de Q =", c2)
    print("P2(-1) =", r2)
    print("Con polyval =", rp2)

    # ------------------
    # Ejercicio 2 (gráficas)
    # ------------------
    print("\nEjercicio 2")
    P = np.array([1., -1, 2, -3, 5, -2])
    R = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    x = np.linspace(-1, 1)

    yP = HornerV(x, P)
    plt.figure()
    plt.plot(x, yP, label='P(x) por HornerV')
    plt.plot(x, 0 * x, 'k')
    plt.title('Polinomio P')
    plt.legend()

    yR = HornerV(x, R)
    plt.figure()
    plt.plot(x, yR, label='R(x) por HornerV')
    plt.plot(x, 0 * x, 'k')
    plt.title('Polinomio R')
    plt.legend()

    # ------------------
    # Ejercicio 3
    # ------------------
    print("\nEjercicio 3")
    x0 = 1.0
    restos_P, derivs_P = dersuc_punto(x0, P)
    print("Restos de dividir P una y otra vez por (x - x0)", restos_P)

    x1 = -1.0
    restos_R, derivs_R = dersuc_punto(x1, R)
    print("Restos de dividir R una y otra vez por (x - x1)", restos_R)

    print("Derivadas sucesivas de P en x0 = 1", derivs_P)
    print("Derivadas sucesivas de R en x1 = -1", derivs_R)

    # ------------------
    # Ejercicio 4
    # ------------------
    print("\nEjercicio 4 (a)")
    for m in (6, 18, 20):
        print(f"Divisores de {m}", divisores(m))

    print("\nEjercicio 4 (b)")
    p0_4 = np.array([-1., 0, 1])
    p1_4 = np.array([8., -6, -3, 1])
    p2_4 = np.array([15., -2, -16, 2, 1])
    p3_4 = np.array([60., 53, -13, -5, 1])
    p4_4 = np.array([490., 343, -206, -56, 4, 1])

    print("Raíces de p0", raices_simples(p0_4))
    print("Raíces de p1", raices_simples(p1_4))
    print("Raíces de p2", raices_simples(p2_4))
    print("Raíces de p3", raices_simples(p3_4))
    print("Raíces de p4", raices_simples(p4_4))

    # ------------------
    # Ejercicio 5
    # ------------------
    print("\nEjercicio 5")
    p1_5 = np.array([8., -22, 17, 1, -5, 1])
    p2_5 = np.array([-135., 378, -369, 140, -9, -6, 1])
    p3_5 = np.array([96., 320, 366, 135, -30, -24, 0, 1])
    p4_5 = np.array([280., 156, -350, -59, 148, -26, -6, 1])

    print("Raíces de p1", raices_multiples(p1_5))
    print("Raíces de p2", raices_multiples(p2_5))
    print("Raíces de p3", raices_multiples(p3_5))
    print("Raíces de p4", raices_multiples(p4_5))

    # ------------------
    # Ejercicio 6
    # ------------------
    print("\nEjercicio 6")
    p0_6 = np.array([1., 2, 1])
    x0_6 = np.array([1., -1])
    y0_6 = hornerVect(x0_6, p0_6)
    print("y =", y0_6)

    # Comparación de tiempos entre HornerV (no vectorizado) y hornerVect
    x_big = np.linspace(-1, 1, 1_000_000)

    t0 = time.time()
    _ = HornerV(x_big, P)
    t1 = time.time()
    _ = hornerVect(x_big, P)
    t2 = time.time()

    print("Tiempo sin vectorización =", t1 - t0)
    print("Tiempo con vectorización =", t2 - t1)

    # Gráficas con hornerVect para P y R
    yP_vec = hornerVect(x, P)
    plt.figure()
    plt.plot(x, yP_vec, label='P(x) por hornerVect')
    plt.plot(x, 0 * x, 'k')
    plt.title('Polinomio P (vectorizado)')
    plt.legend()

    yR_vec = hornerVect(x, R)
    plt.figure()
    plt.plot(x, yR_vec, label='R(x) por hornerVect')
    plt.plot(x, 0 * x, 'k')
    plt.title('Polinomio R (vectorizado)')
    plt.legend()

    # ------------------
    # Ejercicio 7
    # ------------------
    print("\nEjercicio 7")
    x_7 = np.linspace(0, 1)
    Y = derivadasSuc(x_7, P)

    # Dibujar P, P' y P'' (columnas 0, 1, 2 de Y)
    plt.figure()
    plt.plot(x_7, Y[:, 0], label='P')
    plt.plot(x_7, Y[:, 1], label="P'")
    plt.plot(x_7, Y[:, 2], label="P''")
    plt.title('P y sus derivadas primera y segunda')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
