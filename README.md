import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

def f(x):
    return np.sin(x)

# Ejercicio 2
def derivada_numerica_centrada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Ejercicio 3
def trapecio(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])

# Ejercicio A.4
def gauss_legendre(f, a, b, n):

    x_leg, w_leg = leggauss(n)


    t = 0.5 * (x_leg + 1) * (b - a) + a
    integral = 0.5 * (b - a) * np.sum(w_leg * f(t))

    return integral

# Ejercicio A.6
def gauss_legendre_ej6(f, a, b, n):

    return gauss_legendre(f, a, b, n)

# Ejercicio e
def integral_scipy(f, a, b):
    integral, error = quad(f, a, b)
    return integral, error

def main():
    a, b = 0, np.pi
    n = 5
    h = 1e-5

    x0 = np.pi / 4
    derivada = derivada_numerica_centrada(f, x0, h)
    print(f"Ejercicio A.2: Derivada numérica centrada en x = {x0} es {derivada}")

    # Ejercicio A.3
    integral_trapecio = trapecio(f, a, b, n)
    print(f"Ejercicio A.3: Integral con el método del trapecio es {integral_trapecio}")

    # Ejercicio A.4
    integral_gauss_legendre = gauss_legendre(f, a, b, n)
    print(f"Ejercicio A.4: Integral con el método de Gauss-Legendre es {integral_gauss_legendre}")

    # Ejercicio A.6
    integral_ej6 = gauss_legendre_ej6(f, 0, np.pi/2, n)
    print(f"Ejercicio A.6: Integral de Gauss-Legendre en [0, pi/2] es {integral_ej6}")

    # Ejercicio e)
    integral_scipy_result, error_scipy = integral_scipy(f, a, b)
    print(f"Ejercicio e): Integral con scipy es {integral_scipy_result}, con error estimado {error_scipy}")

if __name__ == "__main__":
    main()
