# Bruno Kemmer, Junho 2020
# https://cvxopt.org/
# referência: https://web.archive.org/web/20140429090836/http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/


import numpy as np
from cvxopt import matrix
from cvxopt import solvers


# Kernels
def kernel_linear(X, Y, parametros):
    return np.dot(X,Y)

def kernel_polinomial(X, Y, parametros):
    grau = parametros.get('grau', 2)
    escalar = parametros.get('escalar', 1)
    return np.power((np.dot(X, Y) + escalar), grau)

def kernel_rbf(X, Y,  parametros): #gaussiano
    sigma = parametros.get('sigma', 0.5)
    return np.exp(np.power(-np.linalg.norm(X-Y), 2) / (2 * (np.power(sigma, 2))))

class SVM(object):
    """ Classe SVM - Support Vector Machine
        
        Parâmetros
        ----------
        kernel: {linear, polinomial, rbf}
        grau: utilizado somente no kernel polinomial (grau do polinômio), padrão=2
        escalar: utilizado somente no kernel polinomial (escalar somado), padrão=1
        sigma: utilizado somente no kernel rbf (gaussiano), padrão=0.5
        C: penalizador, padrão=0
        limite: limite para ser um vetor de suporte, padrão=1e-5
    """
    def __init__(self, kernel=kernel_linear, C=0, limite=1e-5, **parametros):
        self.kernel = kernel
        self.parametros = parametros
        self.C = np.float(C)
        self.limite = limite

    def fit(self, X, y):
        n_samples, n_dim = np.shape(X)

        # Matriz kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j], self.parametros)
        
        # Resolvendo o problema quadrático
        P = matrix(np.outer(y, y)*K, tc='d')
        q = matrix(np.ones(n_samples)*-1, tc='d')
        G = matrix(np.identity(n_samples)*-1, tc='d')
        h = matrix(np.zeros(n_samples), tc='d')
        A = matrix(y.reshape((1,n_samples)), tc='d')
        b = matrix(0.0)

        # Solucionando o problema QP
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x']) # alphas
        
        # Identificando os vetores de suporte utilizados e separando em 2 índices: classe positiva e negativa
        vetores_suporte = alphas > self.limite
        vetores_suporte_neg = np.logical_and(vetores_suporte, y < 0)
        vetores_suporte_pos = np.logical_and(vetores_suporte, y > 0)
        
        #intercept - valor mediano nos vetores de suporte
        self.b = (-1/2)*(np.max(K[:,vetores_suporte_neg] @ alphas[vetores_suporte_neg]) + np.min(K[:,vetores_suporte_pos] @ alphas[vetores_suporte_pos]))
        
        self.alphas = alphas[vetores_suporte]
        self.K = K[:,vetores_suporte]
        self.y = y[vetores_suporte]

    # def predict(self, X):
    #     pass

    # def project(self, X):
    #     if self.w is not None:
    #         return np.dot(X, self.w) + self.b
    #     else:
    #         y_predict = np.zeros(len(X))
    #         for i in range(len(X)):
    #             s = 0
    #             for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
    #                 s += a * sv_y * self.kernel(X[i], sv)
    #             y_predict[i] = s
    #         return y_predict + self.b

    # def predict(self, X):
    #     return np.sign(self.project(X))


if __name__ == '__main__':

    # testando
    X = np.array([
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1]
            ])

    y_xor = np.array([-1, 1, 1, -1])
    y_and = np.array([-1, -1, -1, 1])

    svm_clf = SVM(kernel=kernel_linear)
    svm_clf.fit(X, y_and)
    print(svm_clf.alphas)

    svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1)
    svm_clf.fit(X, y_xor)
    print(svm_clf.alphas)

