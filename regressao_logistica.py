# Regressão logistica

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import softmax

def logit(x):
    "Função logística"
    return np.exp(x)/(1+np.exp(x))

def plot_logit(lim_inf=-10, lim_sup=10):
    x = np.linspace(lim_inf, lim_sup)
    y = logit(x)
    plt.plot(x,y)
    plt.show()

def inicializa_w(d, seed=42):
    "Inicializa o vetor de pesos de forma aleatória"
    np.random.seed(seed)
    return np.random.rand(d)

def inicializa_w_multi(atributos, classes, seed=42):
    "Inicializa o vetor de pesos de forma aleatória"
    np.random.seed(seed)
    return np.random.rand(atributos, classes)

def grad_erro(X, y, w, N):
    """ Cálculo do gradiente da função de erro
    
    Arguments:
        X {matriz} -- matriz de exemplos
        y {Vetor} -- vetor de saída da classe
        w {matriz} -- Última matriz de pesos obtida
        N {inteiro} -- Quantidade de elementos de X
    
    Returns:
        float -- Gradiente da função de erro 
    """
    acc = 0
    for i in range(0,N):
        acc += (y[i]*X[i]) / (1 + np.exp(y[i]*np.dot(w, X[i])))
    return -(1/N)*acc

def erro(X, y, w, N):
    """ Calcula o erro para a matriz atual
    
    Arguments:
        X {matriz} -- Matriz de exemplos
        y {Vetor} -- Vetor de saída da classe
        w {matriz} -- Última matriz de pesos obtida
        N {inteiro} -- Quantidade de elementos de X
    
    Returns:
        float -- Erro calculado
    """
    val_erro = 0
    for i in range(0,N):
        val_erro += np.log(1+np.exp(-y[i]*np.dot(w, X[i])))
    return (1/N)*val_erro

def regressao_logistica(X, y, taxa_aprendizado, max_iteracoes=100, tolerancia=1e-5):
    """ Regressão logística
    
    Arguments:
        X {matriz} -- Matriz de treino
        y {Vetor} -- Classe
        taxa_aprendizado {float} -- taxa de aprendizado
    
    Keyword Arguments:
        max_iteracoes {int} -- Máximo de iterações permitido (default: {100})
        tolerancia {float} -- Tolerância máxima (default: {1e-7})
    
    Returns:
        matriz -- Matriz de pesos W treinada
    """
    X = np.column_stack((np.ones((len(X),1)), X))
    N, d = np.shape(X)

    # Inicializa a matriz de pesos em W_0
    w_anterior = inicializa_w(d)
    
    erros = []
    for iteracao in range(0, max_iteracoes):
        grad = grad_erro(X, y, w_anterior, N)
        w = w_anterior - taxa_aprendizado*grad
        w_anterior = w
        erros.append(erro(X, y, w, N))
        if np.linalg.norm(grad) < tolerancia:
            return w, erros
    return w, erros
  
# def softmax(s):
#     return (np.exp(s.T) / np.sum(np.exp(s), axis=1)).T

def regressao_logistica_multiclasse(X, y, taxa_aprendizado, max_iteracoes=100, tolerancia=1e-5):
    """ Regressão logística multiclasse
    
    Arguments:
        X {matriz} -- Matriz de treino
        y {Vetor} -- Classe
        taxa_aprendizado {float} -- taxa de aprendizado
    
    Keyword Arguments:
        max_iteracoes {int} -- Máximo de iterações permitido (default: {100})
        tolerancia {float} -- Tolerância máxima (default: {1e-7})
    
    Returns:
        matriz -- Matriz de pesos W treinada
    """
    X = np.column_stack((np.ones((len(X),1)), X))
    N, d = np.shape(X)

    _, classes = np.shape(y)

    # Inicializa a matriz de pesos em W_0
    w_anterior = inicializa_w_multi(d, classes)
    
    # erros = []
    grad = 0
    for iteracao in range(0, max_iteracoes):
        for i in range(0, N):
            xi = np.expand_dims(X[i], axis=1)
            yi = np.expand_dims(y[i], axis=1)
            grad -= np.dot(xi, (softmax(np.dot(w_anterior.T, xi), axis=0) - yi).T)

        grad = (-1/N) * grad
        w = w_anterior - taxa_aprendizado*grad
        w_anterior = w
        # erros.append(erro(X, y, w, N))
        if np.linalg.norm(grad) < tolerancia:
            return w
    return w

def preditor_logistico(X, w):
    """ Preditor para regressão logística
    
    Arguments:
        X {Matriz} -- Matriz de entrada
        w {Matriz} -- Matriz de pesos (W) previamente calculada
    
    Returns:
        Vetor -- y_hat - Vetor com as classes preditas
    """
    X = np.column_stack((np.ones((len(X),1)), X))
    y_hat = logit(np.dot(X,w))
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = -1
    return y_hat 

def preditor_logistico_multiclasse(X, w):
    """ Preditor para regressão logística multiclasse
    
    Arguments:
        X {Matriz} -- Matriz de entrada
        w {Matriz} -- Matriz de pesos (W) previamente calculada
    
    Returns:
        Vetor -- y_hat - Vetor com as classes preditas
    """
    X = np.column_stack((np.ones((len(X),1)), X))
    y_hat = softmax(np.dot(w.T, X.T), axis=0).T
    y_hat = y_hat.argmax(axis=1)
    return y_hat 
