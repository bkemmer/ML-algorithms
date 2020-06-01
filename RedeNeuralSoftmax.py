import numpy as np
import matplotlib.pyplot as plt

def softmax(A, axis=1):
    expA = np.exp(A)
    return expA / expA.sum(axis=axis, keepdims=True)

def redeNeuralSoftmax(X, y, taxa_aprendizado=0.1, max_iteracoes=15000, custo_min=1e-2, plot=True):
    
    custo = []
    custo_ = np.inf
    
    N, d = np.shape(X)
    n_classes = y.shape[1]

    W = np.random.randn(d+1, n_classes)*0.01
    X = np.concatenate([X, np.ones((N, 1))], axis=1)

    i = 0
    while ((custo_ > custo_min) and (i < max_iteracoes)):
        # forward pass
        Z = X @ W
        y_hat = softmax(Z, axis=1)

        # backpropagation
        dJdW = (1/N)*np.dot(X.T, (y_hat - y))
        W -= taxa_aprendizado * dJdW

        custo_ = np.sum(-y * np.log(y_hat))
        custo.append(custo_)

        if plot:
            if i % 100 == 0:
                print('{}: {}'.format(i, custo_))
        i += 1

    print('Época final: {}\nCusto final: {}'.format(i, custo_))

    return W, custo

def preditorNeuralSoftmax(X_test, W):
    N, d = np.shape(X_test)
    X_test = np.concatenate([X_test, np.ones((N, 1))], axis=1)
    Z = X_test @ W
    y_hat = softmax(Z, axis=1)
    return np.argmax(y_hat, axis=1)
    
if __name__ == '__main__':

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    y = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ])

    epocas = 5000
    t = 0.1
    W, custo = redeNeuralSoftmax(X, y, taxa_aprendizado=t, max_iteracoes=epocas, custo_min=1e-3)
    plt.title('J (Entropia Cruzada com softmax) - função NOR e OR\nt={} e {} epocas'.format(t, epocas))
    plt.ylabel('J')
    plt.xlabel('Épocas')
    plt.plot(custo)
    plt.savefig('./imgs/atividade6_entropia_cruzada_softmax.png')
    plt.show()
