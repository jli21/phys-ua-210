import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

def f(x, a):
    return 1 / np.sqrt(a**4 - x**4)

def gaussian_quadrature(f, a, b, N=20):
    xi, w = leggauss(N)
    integral_approximation = 0.5 * (b - a) * np.sum(w * f(0.5 * (b - a) * xi + 0.5 * (a + b)))
    
    return integral_approximation

amplitudes = np.linspace(0.01, 2, 200)
periods = [np.sqrt(8) * gaussian_quadrature(lambda x: f(x, a), 0, a) for a in amplitudes]

plt.figure(figsize=(10, 6))
plt.plot(amplitudes, periods)
plt.xlabel('Amplitude (a)')
plt.ylabel('Period (t)')
plt.title('Period of the Oscillator vs. Amplitude')
plt.legend()
plt.grid(True)
plt.show()