from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

sigma = 10
r = 28
b = 8/3

def lorenz_system(state, t, sigma, r, b):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

init_cond = [0, 1, 0]
t = np.linspace(0, 50, 10000)  

# TODO: PART A
solution = odeint(lorenz_system, init_cond, t, args=(sigma, r, b))
x, y, z = solution.T

plt.figure(figsize=(10, 6))
plt.plot(t, y, label='y(t)')
plt.title('Lorenz System: y as a Function of Time')
plt.xlabel('Time')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

# TODO: PART B
plt.figure(figsize=(10, 6))
plt.plot(x, z, label='z(x)', color='green')
plt.title('Lorenz System: z as a Function of x (Strange Attractor)')
plt.xlabel('x')
plt.ylabel('z')
plt.grid(True)
plt.legend()
plt.show()  