import numpy as np
import time

def line_sum(i, j, k): 
    return (abs(i**2 + j**2 + k**2))**0.5

def madelung_constant_0(L):
    total = 0.0
    
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                if i == j == k == 0:
                    continue
                else:
                    sign = (1) if (i + j + k) % 2 else -1
                    total += sign / line_sum(i, j ,k)
                    
    return total

def madelung_constant_1(L):
    i, j, k = np.mgrid[-L:L+1, -L:L+1, -L:L+1]
    mask = ~((i == 0) & (j == 0) & (k == 0))
    
    values = np.where((i + j + k) % 2 == 0, -1, 1) / line_sum(i, j ,k)
    total = np.sum(values[mask])
    
    return total

L = 100

start_time = time.time()
M0 = madelung_constant_0(L)
end_time = time.time()
execution_time_madelung_constant_0 = end_time - start_time

start_time = time.time()
M1 = madelung_constant_1(L)
end_time = time.time()
execution_time_madelung_constant_1 = end_time - start_time

print(f"The Madelung constant for L = {L} using madelung_constant_0 is approximately: {M0:.10f}")
print(f"The Madelung constant for L = {L} using madelung_constant_1 is approximately: {M1:.10f}")

print(f"Execution time of madelung_constant_0: {execution_time_madelung_constant_0:.6f} seconds")
print(f"Execution time of madelung_constant_1: {execution_time_madelung_constant_1:.6f} seconds")
