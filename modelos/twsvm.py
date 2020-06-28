import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

import cvxopt
import cvxopt.solvers

# TWSVM
def objective(alpha, ones, M):
        # Optimization
        # M: R*(S.T*S)^(-1)*R.T
        return -(ones.T @ alpha) + (1/2)*(alpha.T @ M @ alpha)

def plotTW(plano_1, plano_2, y_teste, idx):
    import seaborn as sns 
    df = pd.DataFrame(dict(
                        plano_1=np.abs(plano_1[idx]), 
                        plano_2=np.abs(plano_2[idx]), 
                        y_teste=np.where(y_teste[idx]>0, '+1', '-1')
                        ))
    sns.scatterplot(data=df ,x='plano_1', y='plano_2', hue='y_teste', alpha=0.9)
    tmp = np.linspace(np.min(np.abs(plano_1[idx])),np.max(np.abs(plano_1[idx])),100)
    plt.plot(tmp, tmp)
    plt.title('TW-SVWM: Pontos entre os planos')
    plt.xlabel('plano_1: valor absoluto')
    plt.ylabel('plano_2: valor absoluto')
    plt.savefig('./svm_tmp.png')
    plt.show()

def twsvm(X, y, kernel, parametros, eps=1e-16, C_1 = 100, C_2 = 100):

    N, d = np.shape(X)
    # print((N, d))

    A = X[y>0]
    B = X[y<=0]
    C = np.concatenate([A,B])

    # K_A = kernel_pol(A, C, pol=2, escalar=1)
    K_A = kernel(A, C, parametros)
    n_a = A.shape[0]
    e_1 = np.ones((n_a,1))
    S = np.concatenate((K_A, e_1), axis=1)

    TMP_A = (S.T @ S)
    TMP1 = TMP_A + eps*np.identity(TMP_A.shape[0])

    K_B = kernel(B, C, parametros)
    n_b = B.shape[0]
    e_2 = np.ones((n_b,1))
    R = np.concatenate((K_B, e_2), axis=1)


    alpha_inicial = np.zeros(n_b)
    M = R @ np.linalg.pinv(TMP1) @ R.T

    # print(objective(alpha_inicial, e_2, M))

    bound = (0, C_1)
    alpha_bounds = [(bound)]*len(alpha_inicial)
    # const1 = {'type': 'ineq', 'fun': }
    sol = minimize(objective, x0=alpha_inicial, args=(e_2, M), method='SLSQP', bounds=alpha_bounds)
    # print('Soluções alpha:')
    alphas = sol.x
    # print(alphas)

    # print('Valor em alpha*:')
    # print(sol.fun)

    z_1 = -np.linalg.pinv(TMP1) @ R.T @ alphas
    # print(z_1)

    # segundo plano
    L = np.concatenate((K_A, e_1), axis=1)
    N = np.concatenate((K_B, e_2), axis=1)

    TMP_B = (N.T @ N)
    TMP2 = TMP_B + eps*np.identity(TMP_B.shape[0])
    M = L @ np.linalg.pinv(TMP2) @ L.T
    
    #otimização 2 plano
    gamma_inicial = np.zeros(n_a)
    # print(objective(gamma_inicial, e_1, M))
    bound = (0, C_2)
    gamma_bounds = [(bound)]*len(gamma_inicial)
    sol = minimize(objective, x0=gamma_inicial, args=(e_1, M), method='SLSQP', bounds=gamma_bounds)
    # print('Soluções gamma:')
    gammas = sol.x
    # print(gammas)

    # print('Valor em gamma*:')
    # print(sol.fun)

    z_2 = np.linalg.pinv(TMP2) @ L.T @ gammas
    # print(z_2)

    return z_1, z_2

def preditor_twsvm(X_teste, X_treino, y_treino, kernel, parametros, z1, z2, div=False):
    """ Preditor TW-SVM

    Arguments:
        X_test {Matriz} -- Matriz de teste
        X_treino {Matriz} -- Matriz com dados de treino utilizados
        kernel {Função} -- Função de kernel utilizada
        z1 {Vetor} -- [u1 + b1]  
        z2 {Vetor} -- [u2 + b2]

    Returns:
        Vetor -- Vetor com as predições
    """
    
    A = X_treino[y_treino>0]
    B = X_treino[y_treino<=0]
    C = np.concatenate([A,B])

    K = kernel(X_teste, C, parametros)
    plano_1 = K @ z1[:-1] + z1[-1]
    plano_2 = K @ z2[:-1] + z2[-1]
    if div:
        plano_1 /= np.linalg.norm(z1[:-1])
        plano_2 /= np.linalg.norm(z2[:-1])

    idx_pos = plano_1 >= 0
    idx_neg = plano_2 <= 0
    y_hat = np.where(np.abs(plano_1)>np.abs(plano_2), -1, 1)
    y_hat[idx_pos] = 1
    y_hat[idx_neg] = -1

    return y_hat

if __name__ == '__main__':

    from kernels import kernel_linear, kernel_polinomial, kernel_rbf
    # testando
    X = np.array([
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1]
            ])

    y = np.array([-1, 1, 1, -1])
    parametros = {'kernel_polinomial': {'Grau': 2}, 'kernel_rbf': {'Gamma': 0.5}}
    #kernel_linear
    # kernel_rbf
    for kernel in [kernel_polinomial]:
        parametros_kernel = parametros.get(kernel.__name__, {})
        z_1, z_2 = twsvm(X, y, kernel, parametros_kernel, C_1 = 10, C_2 = 10)
        y_hat = preditor_twsvm(X, X, y, kernel, parametros_kernel, z_1, z_2)
        print(y_hat)
        print(y)
        assert((y_hat == y).all())