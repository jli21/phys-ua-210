import matplotlib.pyplot as plt

def f(x):
    return x * (x - 1)

x = 1
delta = 1e-2

def numerical_derivative(f, x, delta): 
    return (f(x + delta) - f(x)) / delta

analytical_derivative = 2 * x - 1

print("Numerical Derivative:", numerical_derivative(f, x, delta))
print("Analytical Derivative:", analytical_derivative)

delta_val = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
errors = []

for delta in delta_val:
    numerical_derivative_val = numerical_derivative(f, x, delta)
    error = abs(numerical_derivative_val - analytical_derivative)
    errors.append(error)

plt.figure(figsize=(10, 6))
plt.loglog(delta_val, errors, marker='o', linestyle='-')
plt.xlabel('Delta (log scale)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Error vs. Delta for Numerical Derivative')
plt.grid(True)
plt.show()





