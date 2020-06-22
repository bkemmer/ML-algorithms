import numpy as np
import matplotlib.pyplot as plt

def Ymulticlasse(y_train, y_test):
    """ Função recebe dois vetores binários e transforma eles em one-hot-encoding

    Arguments:
        y_train {np.array} -- vetor de classes de treino
        y_test {np.array} -- vetor de classes de treino

    Returns:
        (np.array, np.array) -- vetor de classes de treino e teste
    """
    classes = sorted(set(y_train).union(set(y_test)))
    y_train_multi = np.zeros((len(y_train), len(classes)))
    y_test_multi = np.zeros((len(y_test), len(classes)))
    for i, classe in enumerate(classes):
        y_train_multi[:, i] = np.where(y_train == classe, 1, 0)
        y_test_multi[:, i] = np.where(y_test == classe, 1, 0)
    return y_train_multi, y_test_multi

def softmax(A, axis=1):
    A -= np.max(A)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def redeNeuralSoftmax(X, y, h_size=100, taxa_aprendizado=0.5, max_iteracoes=15000, custo_min=1e-5, plot=True):
    
    custo = []
    custo_ = np.inf

    N, d = np.shape(X)
    # n_classes = y.shape[1]
    n_classes = 3

    W1 = np.random.randn(d, h_size)*0.01
    b1 = np.zeros((1,h_size))
    W2 = np.random.randn(h_size, n_classes)*0.01
    b2 = np.zeros((1,n_classes))

  
    i = 0
    while ((custo_ > custo_min) and (i < max_iteracoes)):
        # forward pass
        # A = sigmoid(np.dot(X, W1) + b1)
        A = np.maximum(0, np.dot(X, W1) + b1)
        y_hat = softmax(np.dot(A, W2) + b2, axis=1)

        # Cálculo do custo
        custo_ = (1/N)*np.sum(-y * np.log(y_hat))
        custo.append(custo_)
        if plot and i % 100 == 0:
            print('{}: {:.4}'.format(i, custo_))
        i += 1

        # backpropagation
        dJ = (1/N)*(y_hat - y)
      
        dW2 = A.T @ dJ
        db2 = np.sum(dJ, axis=0, keepdims=True)
        assert(dW2.shape == W2.shape) #, 'dW2 com shape diferente de W2')
        assert(db2.shape == b2.shape) #, 'db2 com shape diferente de b2')

        dA = dJ @ W2.T # derivada do custo
        # FIXME: 
        # dA = (1-A)*A
        dA[A<=0] = 0

        # dW1 = dA * dW1 # derivada sigmoid
        dW1 = X.T @ dA # derivada da parte linear
        db1 = np.sum(dA, axis=0, keepdims=True)

        assert(W1.shape == W1.shape) #, 'dW1 com shape diferente de W1')
        assert(db1.shape == b1.shape) #, 'db1 com shape diferente de b1')
        
        W2 -= taxa_aprendizado * dW2
        b2 -= taxa_aprendizado * db2
        W1 -= taxa_aprendizado * dW1
        b1 -= taxa_aprendizado * db1



    print('Época final: {}\nCusto final: {}'.format(i, custo_))

    return W1, b1, W2, b2, custo

# def preditorNeuralSoftmax(X_test, W1, b1, W2, b2):
#     N, d = np.shape(X_test)
#     Z = X_test @ W1 + b1
#     A = sigmoid(Z @ W2 + b2)
#     y_hat = softmax(A, axis=1)
#     return np.argmax(y_hat, axis=1)

def preditorNeuralSoftmax(X_test, W1, b1, W2, b2):
    N, d = np.shape(X_test)
    A = np.maximum(0, np.dot(X_test, W1) + b1)
    Z = np.dot(A, W2) + b2
    # y_hat = softmax(Z)
    return np.argmax(Z, axis=1)

if __name__ == '__main__':

    # X = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1],
    # ])

    # y = np.array([
    #     [1, 0],
    #     [0, 1],
    #     [0, 1],
    #     [0, 1],
    # ])

    # epocas = 5000
    # t = 0.1
    # W1, b1, W2, b2, custo = redeNeuralSoftmax(X, y, taxa_aprendizado=t, max_iteracoes=epocas, custo_min=1e-3)
    

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
    W1, b1, W2, b2, custo = redeNeuralSoftmax(X, y, h_size=100, taxa_aprendizado=1, max_iteracoes=10000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmax(X, W1, b1, W2, b2)

    print('training accuracy: %.2f' % (np.mean(y_old == y_hat)))
    a = 1