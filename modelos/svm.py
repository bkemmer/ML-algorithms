# Bruno Kemmer, Junho 2020
# https://cvxopt.org/
# referência: https://web.archive.org/web/20140429090836/http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/


import numpy as np
from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False

class SVM(object):
    """ Classe SVM - Support Vector Machine
        
        Parâmetros
        ----------
        kernel: {linear, polinomial, rbf}
        grau: utilizado somente no kernel polinomial (grau do polinômio), padrão=2
        escalar: utilizado somente no kernel polinomial (escalar somado), padrão=1
        gamma: utilizado somente no kernel rbf (gaussiano), padrão=0.5
        C: penalizador, padrão=0
        limite: limite para ser um vetor de suporte, padrão=1e-5
    """
    def __init__(self, kernel, C=0, limite=1e-5, **parametros):
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

        C = self.C
        if int(C) == 0:
            G = matrix(np.identity(n_samples)*-1, tc='d')
            h = matrix(np.zeros(n_samples), tc='d')
        else:
            G_aux = np.vstack((np.identity(n_samples)*-1, np.identity(n_samples)))
            G = matrix(G_aux, tc='d')
            h_aux = np.hstack((np.zeros(n_samples), np.ones(n_samples)*C))
            h = matrix(h_aux, tc='d')

        A = matrix(y.reshape((1,n_samples)), tc='d')
        b = matrix(0.0)

        # Solucionando o problema QP
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x']) # alphas
        
        # Identificando os vetores de suporte utilizados e separando em 2 índices: classe positiva e negativa
        vetores_suporte = alphas > self.limite
        idx_ = np.arange(len(alphas))[vetores_suporte]
        # if int(C) != 0:
        #     vetores_suporte_not = np.logical_not(alphas > C - self.limite)
        #     vetores_suporte = np.logical_and(vetores_suporte, vetores_suporte_not)

        #filtrando somente os vetores de suporte
        alphas = alphas[vetores_suporte]
        X = X[vetores_suporte]
        y = y[vetores_suporte]
        K = K[:,vetores_suporte][vetores_suporte]

        #intercept - valor médio nos vetores de suporte
        w = alphas * y
        y_pred_svm = K @ w
        b = np.mean(y - y_pred_svm)

        self.alphas = alphas
        self.idx_ = idx_
        self.y_vetores_suporte = y
        self.X_vetores_suporte = X
        self.b = b

    def predict(self, X):
        return np.sign(np.sum(self.alphas * self.y_vetores_suporte * 
                                self.kernel(X, self.X_vetores_suporte, self.parametros), axis = 1) + self.b)

def teste():

    # testando
    from kernels import kernel_linear, kernel_polinomial, kernel_rbf
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
    y_hat = svm_clf.predict(X)
    assert((y_hat == y_and).all())
    # print(y_hat)
    # print(y_and)

    svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1)
    svm_clf.fit(X, y_and)
    y_hat = svm_clf.predict(X)
    assert((y_hat == y_and).all())
    # print(y_and)
    # print(y_hat)

    svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1)
    svm_clf.fit(X, y_xor)
    y_hat = svm_clf.predict(X)
    assert((y_hat == y_xor).all())
    # print(y_hat)
    # print(y_xor)



if __name__ == '__main__':
    teste()