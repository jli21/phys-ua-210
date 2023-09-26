import random
import numpy as np
import matplotlib.pyplot as plt

def P(t, tau):
    return 2**(-t/tau) * np.log(2)/tau

def unif(tau, x):
    return (-1/np.log(2)/tau) * np.log(1-x)