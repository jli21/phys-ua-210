import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import pi, log, sqrt, factorial

def integrand(x, a):
    return x**(a-1) * np.exp(-x)

x_values = np.linspace(0, 10, 400)

plt.figure(figsize=(10, 7))

for a in [1, 2, 3, 4, 5]:
    plt.plot(x_values, integrand(x_values, a), label=f'a = {a}')

plt.title("Integrand $x^{a-1} e^{-x}$ for values of $a$")
plt.xlabel("x")
plt.legend()
plt.grid(True)
plt.show()

def gamma(a):
    result, _ = quad(transformed_integrand, 0, 1, args=(a,))
    return result

def transformed_integrand(z, a):
    c = a - 1
    x = z / (1 - z)
    
    integrand_value = exp((a-1) * log(x) - x)
    jacobian = 1 / (1 - z)**2
    
    return integrand_value * jacobian

gamma_ = gamma(3/2)
print("gamma value for 3/2 is : {}".format(gamma_))

gamma_values = [gamma(a) for a in [3, 6, 10]]

factorial_values = [factorial(a-1) for a in [3, 6, 10]]

print("gamma values for 3, 6, 10 are : {} \nfactorial values for 3, 6, 10 are {}".format(gamma_values, factorial_values))
