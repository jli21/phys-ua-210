import numpy as np
import time
import matplotlib.pyplot as plt

def explicit_matrix_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k] * B[k,j]
    return C

sizes = [10, 30, 50, 70]  
explicit_times = []
dot_times = []

for N in sizes:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    
    start_time = time.time()
    explicit_matrix_multiply(A, B)
    end_time = time.time()
    explicit_times.append(end_time - start_time)
    
    start_time = time.time()
    np.dot(A, B)
    end_time = time.time()
    dot_times.append(end_time - start_time)

plt.plot(sizes, explicit_times, label='Explicit Function', marker='o')
plt.plot(sizes, dot_times, label='Dot Method', marker='o')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.title('Time Complexity of Matrix Multiplication')
plt.grid(True)
plt.show()

explicit_times_normalized = [x / y**3 for x, y, in zip(explicit_times, sizes)]
plt.plot(sizes, explicit_times_normalized, marker='o')
plt.xlabel('Matrix Size (N)')
plt.ylabel('t/N^3 (1e-7 seconds)')
plt.title('Normalized Time Complexity of Matrix Multiplication')
plt.grid(True)
plt.show()