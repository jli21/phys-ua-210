import numpy as np
import matplotlib.pyplot as plt

mean = 0
std_dev = 3

x = np.linspace(-10, 10, 1000)  
fig = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std_dev**2))

plt.plot(x, fig, label=f'Mean={mean}, Std Dev={std_dev}')
plt.title('Gaussian Distribution')
plt.xlabel("X")
plt.ylabel("Y")

plt.grid(True)
plt.show()
