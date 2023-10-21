import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ps-5/signal.dat', delimiter = "|")

time = df[df.columns[1]]
signal = df[df.columns[2]]

plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='blue', s=20)
plt.title('Signal vs Time')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.grid(True)
plt.tight_layout()
plt.show()

def svd_fit(order, x, y):

    A = np.vander(x, order + 1)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    S_inv = np.diag(1 / S)
    A_pseudo_inv = Vt.T @ S_inv @ U.T
    coefficients = A_pseudo_inv @ y
    cond_number = np.linalg.cond(A)

    return coefficients, cond_number

coefficients, cond_number = svd_fit(3, time, signal)
fit_values = np.polyval(coefficients, time)

plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='blue', s=20, label="Data Points")
plt.scatter(time, fit_values, color='red', s=20, label=f"3rd Order Polynomial Fit")
plt.title('Signal vs Time with Polynomial Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = signal - fit_values

plt.figure(figsize=(10, 6))
plt.scatter(time, residuals, color='blue', s=20)
plt.axhline(0, color='black', linewidth=1)  
plt.title('Residuals vs Time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

mean_residual, std_residual

coefficients, cond_number = svd_fit(7, time, signal)
fit_values = np.polyval(coefficients, time)

plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='blue', s=20, label="Data Points")
plt.scatter(time, fit_values, color='red', s=20, label=f"7th Order Polynomial Fit")
plt.title('Signal vs Time with Polynomial Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

orders = list(range(4, 21))
condition_numbers = []
fit_values_list = []

for order in orders:
    coefficients, cond_num = svd_fit(order, time, signal)
    fit_values_list.append(np.polyval(coefficients, time))
    condition_numbers.append(cond_num)

condition_numbers