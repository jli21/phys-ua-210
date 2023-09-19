import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(N, max_iter=1000, threshold=2.0):
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    
    C = x[:, np.newaxis] + 1j * y
    
    Z = np.zeros_like(C)
    image = np.ones((N, N, 3))
    
    for i in range(max_iter):
        mask = np.abs(Z) < threshold
        Z[mask] = Z[mask] ** 2 + C[mask]
        
        escape_mask = np.abs(Z) >= threshold
        image[escape_mask] = [0, 0, 0]  
    
    return image

N = 1000
image = mandelbrot(N)
plt.imshow(image, extent=[-2, 2, -2, 2])
plt.show()