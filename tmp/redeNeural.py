# Atividade 6 - Bruno Kemmer N5910474

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# import sklearn.datasets

np.random.seed(42)

def crossEntropyBinaria(y, yHat):
    m = y.shape[0]
    custo = (-1/m)*np.sum(y*np.log(yHat)+(1-y)*np.log(1-yHat))
    custo = np.squeeze(custo)
    return custo

def crossEntropyMulti(y, yHat):
    return np.sum(-y * np.log(yHat))

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoidDerivada(x):
    # input x: sigmoid
    return x*(1 - x)
    
# TODO:
# def softmax_estavel(x, axis):
#     # \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#     # fazendo a softmax ficar estável
#     exps = np.exp(X - np.max(X))
#     return exps / np.sum(exps, axis=axis, keepdims=True)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def derivadaSoftmax(x):
    # input x: softmax vector
    return x*(1 - x)

def dJdyHatSigmoid(y, yHat):
    """ Derivada da função de custo por yHat (resultado do foward pass)"""
    m = y.shape[0]
    return (1/m)*(np.divide(-y, yHat) + np.divide(1 - y, 1 - yHat))

def dJdyHatSoftmax(y, yHat):
    return yHat - y
       
def inicializacao_pesos(nd, nc=1):
    """ Função que inicializa os pesos W = ()

    Arguments:
        nd {int} -- Quantidade da dimensoes
        nc {int} -- Quantidade de classes

    Returns:
        W -- Matriz 
    """
    return np.random.randn(nd, nc)*0.01


def redeNeuralSigmoid(X, y, taxa_aprendizado=0.1, max_iteracoes=5000, custo_min=1e-5, plot=True):
    # número de dimensões + 1 (bias)
    nd = X.shape[1]+1
    W = inicializacao_pesos(nd, 1)
    # caso de uma camada oculta
    W = np.squeeze(W)

    # Aumentando X para conter w0 aka b
    N = X.shape[0]
    X = np.concatenate([X, np.ones((N, 1))], axis=1)
    J = []
    J_atual = np.inf
    i = 0
    while (J_atual > custo_min) and (i < max_iteracoes):
        i += 1
        # forward propagation
        Z = X @ W
        yHat = sigmoid(Z)
        
        J_atual = crossEntropyBinaria(y=y, yHat=yHat)
        J.append(J_atual)
        if plot:
            if i % 100 == 0:
                print('{}: {}'.format(i, J_atual))
        
        # back-propagation
        dJdy_hat = dJdyHatSigmoid(y=y, yHat=yHat)
        dy_hatdZ = sigmoidDerivada(yHat)
        
        dZdW = X
        dJdW = dJdy_hat * dy_hatdZ @ dZdW

        # atualizando os pesos W
        W -= taxa_aprendizado * dJdW

    return W, J

def redeNeuralSoftmax(X, y, taxa_aprendizado=0.1, max_iteracoes=5000, custo_min=1e-5, plot=True):
    # número de dimensões + 1 (bias)
    nd = X.shape[1]+1
    if len(y.shape) < 2:
        nc = 1
    else:
        nc = y.shape[1]

    W = inicializacao_pesos(nd, nc)

    # Aumentando X para conter w0 aka b
    N = X.shape[0]
    X = np.concatenate([X, np.ones((N, 1))], axis=1)
    J = []
    J_atual = np.inf
    i = 0
    while (J_atual > custo_min) and (i < max_iteracoes):
        i += 1
        # forward propagation
        Z = X @ W

        yHat = softmax(Z)
        
        J_atual = crossEntropyMulti(y=y, yHat=yHat)
        J.append(J_atual)
        if plot:
            if i % 100 == 0:
                print('{}: {}'.format(i, J_atual))
        
        # back-propagation
        dJdyHat = dJdyHatSoftmax(y, yHat)

        dZdW = X
        dJdW = dZdW.T @ dJdyHat

        # atualizando os pesos W
        W -= taxa_aprendizado * dJdW

    return W, J

# Teste com NOR e OR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
# NOR e OR
y = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]
])

y_and = np.array([0, 0, 0, 1])

# W, J = redeNeuralSigmoid(X, y_and)
# plt.plot(J)
# plt.title('J (Entropia Cruzada com sigmoid) - função AND\nt=0.1 5k epocas')
# plt.ylabel('J')
# plt.xlabel('Épocas')
# plt.savefig('./imgs/atividade6_entropia_cruzada_sigmoid.png')
# plt.show()

# W, J = redeNeuralSoftmax(X, y)
# plt.plot(J)
# plt.title('J (Entropia Cruzada com softmax) - função NOR e OR\nt=0.1 5k epocas')
# plt.ylabel('J')
# plt.xlabel('Épocas')
# plt.savefig('./imgs/atividade6_entropia_cruzada_softmax.png')
# plt.show()
# plt.show()
