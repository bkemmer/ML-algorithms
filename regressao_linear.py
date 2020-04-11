# Regressão linear

import numpy as np
import matplotlib.pyplot as plt
from utils import acuracia

def regressao_linear(X, y, lamdba=0):
    """ Cálculo da regressão linear na forma vetorial sem regularização

    Arguments:
        X {Matriz} -- Matriz dos exemplos de entrada já com a coluna com x_0 = 1 adicionada 
        y {Vetor} --  Vetor da classe de saída 
        lamdba {float} -- Fator de regularização
    Returns:
        [W] -- Matriz de pesos 
    """
    # Adicionadno a coluna x_0 que será multiplicada com o viés (bias)
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)
    _, d = np.shape(X)
    return np.linalg.inv(X.T.dot(X) + lamdba*np.identity(d)).dot(X.T).dot(y)

def preditor_linear(w, X):
    """ Preditor da função linear estimada

    Arguments:
        w {Vetor} -- Vetor de pesos
        x {Vetor} -- Vetor com os atributos do exemplo em questão

    Returns:
        classe -- classe estimada
    """
    # Adicionadno a coluna x_0 que será multiplicada com o viés (bias)
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)

    if len(np.shape(w)) < 2:
        y_hat = np.sum(w*X, axis=1)
        return y_hat

    # Caso multi-class
    _, n_classes = np.shape(w)
    y_hat = np.ones((len(X), n_classes))
    for k in range(0, n_classes):
        y_hat[:, k] = np.sum(w[:, k]*X, axis=1)
    return np.argmax(y_hat, axis=1)

def plot_regularizacao(X_train, y_train, X_test, y_test, 
                        limits_min=0, limits_max=100,
                        split_n=100, 
                        output_file_name=None):

    acc_train = []
    acc_test = []
    lamdbas = []
    w_means = []
    for i in range (limits_min, limits_max):
        lamdba = i/split_n

        w = regressao_linear(X_train, y_train, lamdba=lamdba)
        y_hat_test = preditor_linear(w, X_test)
        y_hat_train = preditor_linear(w, X_train)
        
        w_means.append(np.mean(np.abs(w)))
        lamdbas.append(lamdba)
        acc_test.append(acuracia(y_hat_test, y_test, show=False))
        acc_train.append(acuracia(y_hat_train, y_train, show=False))
    
    _, ax1 = plt.subplots(figsize=(15,15))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(lamdbas, acc_train, label="Treino", color='b')
    ln2 = ax1.plot(lamdbas, acc_test, label="Teste", color='g')
    ln3 = ax2.plot(lamdbas, w_means, label="Média dos pesos (w)", color='r')
    ax1.set_xlabel('lambdas (fator de regularização')
    ax1.set_ylabel('Acurácia')
    ax2.set_ylabel('Média dos pesos (w)')
    ax1.set_title('Regressão linear variando o fator de regularização')
    
    # unificando as legendas
    lns= ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(output_file_name)
    plt.show()

