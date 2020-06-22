# SIN 5016 - Bruno Kemmer
import numpy as np
import matplotlib.pyplot as plt

from tests import *

def sigmoid(Z):
    """ Função sigmoid

    Args:
        Z (np.array): Vetor de entrada

    Returns:
        A: Vetor com a função sigmoid aplicada a Z
        cache: contém Z que foi utilizado como entrada, para o backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    assert(A.shape == Z.shape)

    return A, cache

def grad_sigmoid(dA, cache):
    """ Gradiente da função sigmoid para o backpropagation

    Args:
        dA: Gradiente da camada anterior
        cache: Z utilizado

    Returns:
        np.array: Gradiente do custo em relação a Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu(Z):
    """ Função relu

    Args:
        Z (np.array): Vetor de entrada

    Returns:
        A: Vetor com a função relu aplicada a Z
        cache: contém Z que foi utilizado como entrada, para o backpropagation
    """

    A = np.maximum(0,Z)
    cache = Z
    assert(A.shape == Z.shape)
    
    return A, cache

def grad_relu(dA, cache):
    """ Gradiente da função relu
        Quando o valor for maior que 0, "passa ele", caso contrário é zero. 

    Args:
        dA: Gradiente a camada anterior
        cache: Z utilizado no foward pass

    Returns:
        np.array: Gradiente do custo em relação a Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    
    return dZ


def inicializa_camadas(camadas, seed=42):
    """ Inicializa as camadas da rede

    Args:
        camadas_dim (list): lista contendo as dimensões de cada camada da rede

    Returns:
        dict: dicionário contendo os parâmetros da rede "W1", "b1", "W2","b2", ...
    """
    np.random.seed(seed)
    parametros = {}
    n_camadas = len(camadas)
    for i in range(1, n_camadas):
        parametros['W' + str(i)] = np.random.randn(camadas[i], camadas[i-1]) * 0.01
        parametros['b' + str(i)] = np.zeros((camadas[i], 1)) 
    return parametros

def linear_pass(A, W, b):
    """ Passagem linear

    Args:
        A (np.array): ativações da camada anterior
        W (np.array): matriz de pesos: (dimensão da camada atual x dimensão da camada anterior)
        b (np.array): vetor de bias (tamanho da camada atual x 1)

    Returns:
        Z: resultado do feedforward
        cache: tupla contendo (A, W, b) para ser utilizada na próxima camadas
    """
    Z = A @ W  + b
    cache = (A, W, b)
    assert(Z.shape == (A.shape[0], W.shape[1]))

    return Z, cache

def forward_pass_linear(A_anterior, W, b, ativacao):
    """ Etapa de aplicação da função de ativação na etapa de foward pass

    Args:
        A_anterior (np.array): resultado da ativação da camada anterior (tamanho da camada anterior x quantidade de exemplos)
        W (np.array): matriz de pesos 
        b (np.array): bias
        funcao_ativacao (string): String determinando qual função de ativação será utilizada

    Returns:
        A: matriz com o resultado da aplicação da função de ativação
        cache: tupla com (linear_cache, ativacao_cache)
    """
    if ativacao == 'sigmoid':
        Z, linear_cache = linear_pass(A_anterior, W, b)
        A, ativacao_cache = sigmoid(Z)
    elif ativacao == 'relu':
        Z, linear_cache = linear_pass(A_anterior, W, b)
        A, ativacao_cache = relu(Z)

    cache = (linear_cache, ativacao_cache)
    assert(A.shape == (A_anterior.shape[0], W.shape[1]))
    
    return A, cache

def calcule_custo(yHat, y):
    """ Cálculo da função de custo - entropia cruzada

    Args:
        yHat (np.array): Resultado do foward pass, predição do modelo
        y (np.array): classe verdadeira do dataset

    Returns:
        custo: Resultado da função de custo
    """
    n = y.shape[1]
    # custo = np.sum(-y * np.log(yHat))
    custo = (-1/n)*np.sum(y*np.log(yHat)+(1-y)*np.log(1-yHat))
    cost = np.squeeze(custo)
    return custo

def backward_pass_linear(dZ, cache):
    """ Calcula a propagação linear do gradiente em uma única camada

    Args:
        dZ (np.array): Gradiente do custo da parte linear (camada atual)
        cache ([type]): tupla (A_anterior, W, b) da forward propagation da camada atual

    Returns:
        dA_anterior: Gradiente do custo da parte linear (da camada anterior)
        dW: Gradiente do custo em relação a W
        db: Gradiente do custo em relação a b
    """
    A_anterior, W, b = cache
    n = A_anterior.shape[0]
    # FIXME:
    dW = A_anterior.T @ dZ # (1/n) * 
    db = np.sum(dZ, axis=0, keepdims=True) # (1/n) *
    dA_anterior = dZ @ W.T

    assert(dA_anterior.shape == A_anterior.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_anterior, dW, db

def backward_pass_ativacao(dA, cache, ativacao):
    """ Executa a backward propagation na parte da função de ativação.

    Args:
        dA (np.array): Gradiente da camada atual (após ativação).
        cache (tuple): Tupla (linear_cache, ativacao_cache) para deixar o código mais eficiente
        ativacao (string): String que determina qual função de ativação foi utilizada: {'sigmoid', 'relu'}

    Returns:
        dA_anterior: Gradiente do custo em relação a camada anterior
        dW: Gradiente do custo em relação a W
        db: Grandiente do custo em relação a b
    """
    linear_cache, ativacao_cache = cache
    if ativacao == 'relu':
        dZ = grad_relu(dA, ativacao_cache)
        dA_anterior, dW, db = backward_pass_linear(dZ, linear_cache)
    elif ativacao == 'sigmoid':
        dZ = grad_sigmoid(dA, ativacao_cache)
        dA_anterior, dW, db = backward_pass_linear(dZ, linear_cache)
    return dA_anterior, dW, db

def atualiza_parametros(parametros, grads, taxa_aprendizado):
    """ Atualiza os parâmetros usando o algoritmo gradiente descendente

    Args:
        parametros (Dict): Dicionário com os parâmetros
        grads (Dict): Dicionário contendo os gradientes
        taxa_aprendizado (float): taxa de aprendizado

    Returns:
        parametros: Dicionário com os parâmetros atualizados
    """
    L = len(parametros) // 2 # quantidade de camadas
    for i in range(L):
        parametros["W" + str(i+1)] -= taxa_aprendizado*grads["dW" + str(i+1)]
        parametros["b" + str(i+1)] -= taxa_aprendizado*grads["db" + str(i+1)]
    return parametros

def predicao(X, parametros, funcoes):
    pass

if __name__ == "__main__":
    pass

