import numpy as np
import matplotlib.pyplot as plt

tau = 3.053 * 60
mu = np.log(2)/tau

z = np.random.rand(1000)
x = -1/mu * np.log(z)

plt.figure(figsize=(10, 6))
plt.plot(np.sort(x), np.arange(1000, 0, -1))
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Atoms (not decayed)')
plt.title('Decay of 208Tl')
plt.grid(True)
plt.show()
