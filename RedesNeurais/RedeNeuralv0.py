import numpy as np
import matplotlib.pyplot as plt

def softmax(A, axis=1):
    A -= np.max(A)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def redeNeuralSoftmax(X, y, h_size=100, taxa_aprendizado=0.1, max_iteracoes=15000, custo_min=1e-2, plot=True):
    
    custo = []
    custo_ = np.inf

    N, d = np.shape(X)
    n_classes = y.shape[1]

    # d+1 por causa do bias
    W1 = np.random.randn(d+1, h_size)*0.01
    W2 = np.random.randn(h_size, n_classes)*0.01
    X = np.concatenate([X, np.ones((N, 1))], axis=1)

    i = 0
    while ((custo_ > custo_min) and (i < max_iteracoes)):
        # forward pass
        A = sigmoid(X @ W1)
        y_hat = softmax(A @ W2, axis=1)

        # backpropagation
        dJ = (1/N)*(y_hat - y)
        dW2 = A.T @ dJ 

        dA = A*(1-A)
        dW1 = X.T @ dA

        # dJdW = (1/N)*(X.T @ (dJdA * dAdZ))
        
        # dJdW = (1/N) *() X.T @ (dJdA * dAdZ)
        W2 -= taxa_aprendizado * dW2
        W1 -= taxa_aprendizado * dW1
        custo_ = np.sum(-y * np.log(y_hat))
        custo.append(custo_)

        if plot and i % 100 == 0:
            print('{}: {:.4}'.format(i, custo_))
        i += 1

    print('Época final: {}\nCusto final: {}'.format(i, custo_))

    return W1, W2, custo

def preditorNeuralSoftmax(X_test, W):
    N, d = np.shape(X_test)
    X_test = np.concatenate([X_test, np.ones((N, 1))], axis=1)
    Z = X_test @ W
    y_hat = softmax(Z, axis=1)
    return np.argmax(y_hat, axis=1)
    
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
    # W, custo = redeNeuralSoftmax(X, y, taxa_aprendizado=t, max_iteracoes=epocas, custo_min=1e-3)
    # plt.title('J (Entropia Cruzada com softmax) - função NOR e OR\nt={} e {} epocas'.format(t, epocas))
    # plt.ylabel('J')
    # plt.xlabel('Épocas')
    # plt.plot(custo)
    # plt.savefig('./imgs/atividade6_entropia_cruzada_softmax.png')
    # plt.show()
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
    
    y, _ = Ymulticlasse(y, y)
    
    epocas = 5000
    t = 0.1
    W1, W2, custo = redeNeuralSoftmax(X, y, taxa_aprendizado=t, max_iteracoes=epocas, custo_min=1e-3)