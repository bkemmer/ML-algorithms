# Trabalho parcial - SIN 5016 - Aprendizado de máquina
# Bruno Kemmer - NUSP: 5910474

import numpy as np
import matplotlib.pyplot as plt
from RedeNeuralv2 import *

import h5py

def inicializa_parametros(n_d, n_h, n_y, seed=42):
    """ Função para inicializar os parâmetros da rede

    Args:
        n_d (int): Tamanho da camada de entrada (dimensão)
        n_h (int): Tamanho da camada oculta
        n_y (int): Tamanho da camada de saída (quantidade de classes)

    Returns:
        dict: dicionário com os parâmetros:
            W1 -- matriz de pesos (n_d x n_h)
            b1 -- vetor de bias
            W2 -- matriz de pesos (n_h x n_y)
            b2 -- vetor de bias
    """
    np.random.seed(seed)
    W1 = np.random.randn(n_d, n_h)*0.01
    b1 = np.zeros((1,n_h))
    W2 = np.random.randn(n_h, n_y)*0.01
    b2 = np.zeros((1,n_y))

    assert(W1.shape == (n_d, n_h))
    assert(b1.shape == (1,n_h))
    assert(W2.shape == (n_h, n_y))
    assert(b2.shape == (1,n_y))

    return {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}

def RedeNeuralTrabalhoParcial(X, Y, camadas_escondidas, taxa_aprendizado, max_iter=5000, print_custo=False, seed=42):
    """ Rede Neural de 2 camadas 

    Args:
        X (np.array): [description]
        Y (np.array): [description]
        camadas_escondidas (list): Lista de inteiros com a quantidade de neuronios por camada escondida
        taxa_aprendizado (float): Taxa de aprendizado
        max_iter (int, optional): Quantidade máxima de iterações. Defaults to 5000.
        print_custo (bool, optional): Mostrar o custo a cada 100 iterações e mostra o gráfico da função de custo. Defaults to False.
        seed (int, optional): Random State. Defaults to 42
    Returns:
        parametros: Dicionário com os parâmetros da rede treinada
    """
    np.random.seed(seed)
    grads = {}
    custos = []
    n, m = X.shape
    n_classes = Y.shape[1]

    n_h = np.squeeze(camadas_escondidas)
    n_y = n_classes
    parametros = inicializa_parametros(m, n_h, n_y)

    W1 = parametros["W1"]
    b1 = parametros["b1"]
    W2 = parametros["W2"]
    b2 = parametros["b2"]

    for i in range(max_iter):

        # Foward pass
        A1, cache1 = forward_pass_linear(X, W1, b1, ativacao="sigmoid")
        A2, cache2 = forward_pass_linear(A1, W2, b2, ativacao="sigmoid")

        # Calculo do custo
        custo = calcule_custo(A2, Y)

        # Backward propagation
        dA2 = (1/n)*(A2 - Y)
        dA1, dW2, db2 = backward_pass_ativacao(dA2, cache2, ativacao="sigmoid")
        dA0, dW1, db1 = backward_pass_ativacao(dA1, cache1, ativacao="sigmoid")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Atualiza os parâmetros
        parametros = atualiza_parametros(parametros, grads, taxa_aprendizado)

        W1 = parametros["W1"]
        b1 = parametros["b1"]
        W2 = parametros["W2"]
        b2 = parametros["b2"]

        # Mostra o custo a cada 100 iterações
        if print_custo and i % 100 == 0:
            print("Custo após a iterção {}: {}".format(i, np.squeeze(custo)))
            custos.append(custo) 

    if print_custo:
        plt.plot(np.squeeze(custos))
        plt.ylabel('Valor da função de custo')
        plt.xlabel('Iterações')
        plt.title("Taxa de aprendizado =" + str(taxa_aprendizado))
        plt.show()

    print(parametros)
    return parametros


def main():
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
    camadas_escondidas = [3]
    _ = RedeNeuralTrabalhoParcial(X, y, camadas_escondidas, taxa_aprendizado=t, print_custo=True)

if __name__ == '__main__':
    main()
    
    np.random.seed(1)
    def load_data():
        train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    
    epocas = 5000
    t = 0.1
    camadas_escondidas = [n_h]
    parametros = RedeNeuralTrabalhoParcial(train_x, train_y, camadas_escondidas, taxa_aprendizado=t, print_custo=True)