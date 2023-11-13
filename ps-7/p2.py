import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = pd.read_csv("survey.csv")
init_beta = np.array([0.0, 0.0])

def logistic_function(beta, x):
    return 1 / (1 + np.exp(-(beta[0] + beta[1] * x)))

def neg_log_likelihood(beta, x, y):
    p = logistic_function(beta, x)
    epsilon = 1e-5
    p = np.clip(p, epsilon, 1 - epsilon)
    log_likelihood = y * np.log(p) + (1 - y) * np.log(1 - p)
    return -np.sum(log_likelihood)

def gradient(beta, x, y):
    p = logistic_function(beta, x)
    return np.array([np.sum((p - y)), np.sum((p - y) * x)])

def hessian(beta, x, y):
    p = logistic_function(beta, x)
    dp = p * (1 - p)  
    h00 = np.sum(dp)
    h01 = h11 = np.sum(dp * x)
    h10 = h01
    return np.array([[h00, h01], [h10, h11]])

result = minimize(fun=neg_log_likelihood,
                     x0=init_beta,
                     args=(data['age'], data['recognized_it']),
                     method='BFGS',
                     jac=gradient)

if result.success:
    est_beta = result.x
    cov_matrix = np.linalg.inv(hessian(est_beta, data['age'], data['recognized_it']))
else:
    est_beta = None
    cov_matrix = None

plt.scatter(data['age'], data['recognized_it'], alpha=0.5, label='Survey Data')
ages = np.linspace(data['age'].min(), data['age'].max(), 300)
probabilities = logistic_function(est_beta, ages)
plt.plot(ages, probabilities, label='Logistic Regression', color='red')
plt.xlabel('Age')
plt.ylabel('Probability of Recognition')
plt.legend()
plt.grid(True)

plt.show()

est_beta, cov_matrix, result.fun
