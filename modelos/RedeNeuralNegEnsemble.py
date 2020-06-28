import numpy as np
import matplotlib.pyplot as plt

def softmax(A, axis=1):
    A -= np.max(A)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def correlacao_negativa(yHats, y, i, lamdba):
    """ Cálculo da correlação negativa

    Args:
        yHats (np.array): Predições das M redes
        y (np.array): Valor (real) de y
        i (int): índice da rede atual
        lamdba (float): termo regularizador do ensamble [0,1]

    Returns:
        np.array: correlação negativa da rede i
    """
    j = np.arange(len(yHat), dtype=int)
    p_i = (yHat[i] - y)*np.sum(yHat[j!=i]-y, axis=1)
    return lamdba*np.mean(p_i)

def dCustoYhat(yHats, y, i, lamdba):
    """ Cálculo da derivada do Custo em função da predição da rede i

    Args:
        yHats (np.array): Predições das M redes
        y (np.array): Valor (real) de y
        i (int): índice da rede atual
        lamdba (float): termo regularizador do ensamble [0,1]

    Returns:
        np.array: derivada do Custo em função de yHat_i
    """
    f_n = np.mean(yHats)
    return (1-lamdba)*(yHats[i]-y) + lamdba*(f_n-y)

def inicializaRede(d, h_size, n_classes):  # sourcery skip: merge-dict-assign
    parametros = {}
    parametros['W1'] = np.random.randn(d, h_size)*0.01
    parametros['b1'] = np.zeros((1,h_size))
    parametros['W2'] = np.random.randn(h_size, n_classes)*0.01
    parametros['b2'] = np.zeros((1,n_classes))
    return parametros

def foward_pass(X, W1, b1, W2, b2, ativacao):
    # forward pass
    if ativacao == 'sigmoid':
        A = sigmoid(np.dot(X, W1) + b1)
    elif ativacao == 'relu':
        A = np.maximum(0, np.dot(X, W1) + b1)

    y_hat = softmax(np.dot(A, W2) + b2, axis=1)
    return A, y_hat

def backpropagation(y_hat, y, A, W2, b2, X, W1, b1, ativacao):
    N,_ = np.shape(y)
    dJ = (1/N)*(y_hat - y)
    
    dW2 = A.T @ dJ
    db2 = np.sum(dJ, axis=0, keepdims=True)
    assert(dW2.shape == W2.shape) #, 'dW2 com shape diferente de W2')
    assert(db2.shape == b2.shape) #, 'db2 com shape diferente de b2')

    dA = dJ @ W2.T # derivada do custo
    # derivada parte não-linear
    if ativacao == 'sigmoid':
        dA = dA * (1-A)*A
    elif ativacao == 'relu':
        dA[A<=0] = 0

    dW1 = X.T @ dA # derivada da parte linear
    db1 = np.sum(dA, axis=0, keepdims=True)
    assert(W1.shape == W1.shape) #, 'dW1 com shape diferente de W1')
    assert(db1.shape == b1.shape) #, 'db1 com shape diferente de b1')
    
    return dW1, db1, dW2, db2

def atualizaParametros(parametros, dW1, db1, dW2, db2, taxa_aprendizado):
    parametros['W1'] -= taxa_aprendizado * dW1
    parametros['b1'] -= taxa_aprendizado * db1
    parametros['W2'] -= taxa_aprendizado * dW2
    parametros['b2'] -= taxa_aprendizado * db2

    return parametros

def calculaCusto(y_hat, y, custo, i, plot):
    N,_ = np.shape(y)
    custo_ = (1/(2*N))*np.sum((y_hat - y)**2)
    custo.append(custo_)
    if plot and i % 100 == 0:
        print('{}: {:.4}'.format(i, custo_))
    return custo#+ correlacao_negativa()

    #n_redes, lamdba, 
def redeNeuralSoftmax(X, y, h_size=100, ativacao='sigmoid', taxa_aprendizado=0.5, max_iteracoes=15000, custo_min=1e-5, plot=True):
    
    custo = []
    custo_ = np.inf

    N, d = np.shape(X)
    n_classes = y.shape[1]
    parametros = inicializaRede(d, h_size, n_classes)

    i = 0
    while ((custo_ > custo_min) and (i < max_iteracoes)):

        W1, b1, W2, b2 = parametros['W1'], parametros['b1'], parametros['W2'], parametros['b2']
        #forward pass    
        A, y_hat = foward_pass(X, W1, b1, W2, b2, ativacao)
        # Cálculo do custo
        custo = calculaCusto(y_hat, y, custo, i, plot)
        # backpropagation
        dW1, db1, dW2, db2 = backpropagation(y_hat, y, A, W2, b2, X, W1, b1, ativacao)
        #atualiza parâmetros
        parametros = atualizaParametros(parametros, dW1, db1, dW2, db2, taxa_aprendizado)
        i += 1
    print('Época final: {}\nCusto final: {}'.format(i, custo_))

    return W1, b1, W2, b2, custo

def preditorNeuralSoftmax(X_test, W1, b1, W2, b2, ativacao='sigmoid'):
    N, d = np.shape(X_test)
    if ativacao == 'sigmoid':
        A = sigmoid(np.dot(X_test, W1) + b1)
    elif ativacao == 'relu':
        A = np.maximum(0, np.dot(X_test, W1) + b1)
    Z = np.dot(A, W2) + b2
    return np.argmax(Z, axis=1)

def teste():
    from utils import Ymulticlasse
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    y_old = np.copy(y)
    
    y, _ = Ymulticlasse(y, y)
    print("Teste sigmoid:\n")
    W1, b1, W2, b2, custo = redeNeuralSoftmax(X, y, h_size=100, ativacao='sigmoid', taxa_aprendizado=1, max_iteracoes=10000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmax(X, W1, b1, W2, b2, ativacao='sigmoid')
    print('training accuracy: %.2f' % (np.mean(y_old == y_hat)))
    print("\nTeste relu:\n")
    W1, b1, W2, b2, custo = redeNeuralSoftmax(X, y, h_size=100, ativacao='relu', taxa_aprendizado=1, max_iteracoes=10000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmax(X, W1, b1, W2, b2, ativacao='relu')
    print('training accuracy: %.2f' % (np.mean(y_old == y_hat)))

if __name__ == '__main__':
    teste()
