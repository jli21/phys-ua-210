from __future__ import division, print_function
import numpy as np
from vpython import curve, rate, vector

# Function to solve a banded system of linear equations
def banded(Aa, va, up, down):
    A = np.copy(Aa)
    v = np.copy(va)
    N = len(v)

    # Gaussian elimination
    for m in range(N):
        div = A[up, m]
        v[m] /= div
        for k in range(1, down + 1):
            if m + k < N:
                v[m + k] -= A[up + k, m] * v[m]

        for i in range(up):
            j = m + up - i
            if j < N:
                A[i, j] /= div
                for k in range(1, down + 1):
                    A[i + k, j] -= A[up + k, m] * A[i, j]

    # Backsubstitution
    for m in range(N - 2, -1, -1):
        for i in range(up):
            j = m + up - i
            if j < N:
                v[m] -= A[i, j] * v[j]

    return v

# Constants
h = 1e-18 * 10
hbar = 1.0546e-36
L = 1e-8
M = 9.109e-31
N = 1000
a = L / N

# Coefficients for matrices
a1 = 1 + h * hbar / (2 * M * a**2) * 1j
a2 = -h * hbar * 1j / (4 * M * a**2)
b1 = 1 - h * hbar / (2 * M * a**2) * 1j
b2 = h * hbar * 1j / (4 * M * a**2)

# Initial wave function
def ksi0(x):
    x0 = L / 2
    sigma = 1e-10
    k = 5e10
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

ksi = np.zeros(N + 1, complex)
x = np.linspace(0, L, N + 1)
ksi[:] = ksi0(x)
ksi[[0, N]] = 0  # Boundary conditions

# Setting up the matrix A
A = np.empty((3, N), complex)
A[0, :] = a2
A[1, :] = a1
A[2, :] = a2

# Visualization setup
ksi_c = curve()
for xi in x:
    ksi_c.append(vector(xi - L/2, 0, 0))

while True:
    rate(30)  # Control the animation frame rate

    # Compute v and update ksi
    for i in range(20):
        v = b1*ksi[1:N] + b2*(ksi[2:N+1] + ksi[0:N-1])
        ksi[1:N] = banded(A, v, 1, 1)

    # Update the curve points directly
    for i, xi in enumerate(x):
        ksi_c.modify(i, vector(xi - L/2, np.real(ksi[i]) * 1e-9, np.imag(ksi[i]) * 1e-9))