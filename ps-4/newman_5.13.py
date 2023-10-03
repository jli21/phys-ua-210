import matplotlib.pyplot as plt
import numpy as np
from math import factorial, sqrt, exp, pi
from scipy.integrate import quad

def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    H_nminus2 = 1
    H_nminus1 = 2 * x
    H_n = 0
    for i in range(2, n+1):
        H_n = 2 * x * H_nminus1 - 2 * (i-1) * H_nminus2
        H_nminus2, H_nminus1 = H_nminus1, H_n
    return H_n

def psi(n, x):
    func = 1 / sqrt(2**n * factorial(n) * sqrt(pi)) * H(n, x) * exp(-x**2 / 2)
    return func

x_values = np.linspace(-4, 4, 500)

for n in range(4):
    y_values = [psi(n, x) for x in x_values]
    plt.plot(x_values, y_values, label=f'n={n}')
plt.title('Harmonic oscillator wave functions for n=4')
plt.xlabel('x')
plt.ylabel('psi_n(x)')
plt.legend()
plt.grid(True)
plt.show()

x_values = np.linspace(-10, 10, 500)
for n in range(30):
    y_values = [psi(n, x) for x in x_values]
    plt.plot(x_values, y_values, label=f'n={n}')
plt.title('Harmonic oscillator wave functions for n=30')
plt.xlabel('x')
plt.ylabel('psi_n(x)')
plt.grid(True)
plt.show()

def integrand(x, n):
    return x**2 * abs(psi(n, x))**2

def uncertainty(n):
    res, _ = quad(integrand, -np.inf, np.inf, args=(n))
    return sqrt(res)

def uncertainty_gauss_hermite(n, num_points=1000):
    nodes, weights = np.polynomial.hermite.hermgauss(num_points)
    integral = sum(weights[i] * (nodes[i]**2) * abs(psi(n, nodes[i]))**2 for i in range(num_points))
    return sqrt(integral)

result_gh = uncertainty_gauss_hermite(5)
print(f"The uncertainty for n = 5 using Gauss-Hermite quadrature is approximately {result_gh:.2f}")