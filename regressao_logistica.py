# Regressão logistica

import numpy as np
import matplotlib.pyplot as plt

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

def grad_erro(X, y, w):
    """ Cálculo do gradiente da função de erro
    
    Arguments:
        X {matriz} -- matriz de exemplos
        y {[type]} -- vetor de saída da classe
        w {matriz} -- Última matriz de pesos obtida
    
    Returns:
        float -- Gradiente da função de erro 
    """
    acc = 0
    N = len(X)
    for n in range(0,N):
        acc += y[n]*X[n,:] / (1 + np.exp(y[n]*np.dot(w, X[n])))
    return -(1/N)*acc

def regressao_logistica(X, y, taxa_aprendizado, max_iteracoes=100, tolerancia=1e-7):
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
    _, d = np.shape(X)

    # Inicializa a matriz de pesos em W_0
    w_anterior = inicializa_w(d)
    
    tolerancia = np.inf
    for _ in range(0, max_iteracoes):
        w = w_anterior - taxa_aprendizado*grad_erro(X, y, w_anterior)
        w_anterior = w

    return w

<<<<<<< HEAD
# TODO: colocar critério de parada na logistica
        # if np.linalg.norm(w) < tolerancia:
        #     break
  
def preditor_logistico(X, w):
    """ Preditor para regressão logística
    
    Arguments:
        X {Matriz} -- Matriz de entrada
        w {Matriz} -- Matriz de pesos (W) previamente calculada
    
    Returns:
        Vetor -- y_hat - Vetor com as classes preditas
    """
    y_hat = logit(np.sum(w*X, axis=1))
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = -1
    return y_hat 
=======
def preditor_logistico(X, w, corte=0.5):
    """[summary]
    
    Arguments:
        X {[type]} -- [description]
        w {[type]} -- [description]
    
    Keyword Arguments:
        corte {int} -- [description] (default: {0})
    
    Returns:
        [type] -- [description]
    """
    y_hat = logit(np.sum(w*X, axis=1))
    y_hat[y_hat >= corte] = 1
    y_hat[y_hat < corte] = -1
    return y_hat 



            # if np.linalg.norm(w) < tolerancia:
        #     break
>>>>>>> 0ecee76c6cb14f00bc0295261c867c204525c5d3
