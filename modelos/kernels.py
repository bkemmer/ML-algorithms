import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
# Kernels
def kernel_linear(X, Y, parametros):
    return np.dot(X,Y.T)

def kernel_polinomial(X, Y, parametros):
    grau = parametros.get('Grau', 2)
    escalar = parametros.get('Escalar', 1)
    return np.power((np.dot(X, Y.T) + escalar), grau)

def kernel_rbf(X, Y, parametros): #gaussiano
    gamma = parametros.get('Gamma', 0.5)
    # return rbf_kernel(X.reshape(-1, 1), Y.reshape(-1, 1), gamma=gamma)
    return np.exp(np.power(-np.linalg.norm(X-Y), 2) / (2 * (np.power(gamma, 2))))