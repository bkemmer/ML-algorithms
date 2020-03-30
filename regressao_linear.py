# Regressão linear

import numpy as np
import utils

def regressao_linear(X, y):
    """ Cálculo da regressão linear na forma vetorial sem regularização

    Arguments:
        X {Matriz} -- Matriz dos exemplos de entrada já com a coluna com x_0 = 1 adicionada 
        y {Vetor} --  Vetor da classe de saída 

    Returns:
        [W] -- Matriz de pesos 
    """
    return np.dot(np.dot(np.linalg.inv((np.dot(np.transpose(X), X))), np.transpose(X)), y)

def preditor_linear(w, X):
    """ Preditor da função linear estimada

    Arguments:
        w {Vetor} -- Vetor de pesos
        x {Vetor} -- Vetor com os atributos do exemplo em questão

    Returns:
        classe -- classe estimada
    """
    _, n_classes = np.shape(w)
    if n_classes > 2:
        y_hat = np.ones((len(X), n_classes))
        for k in range(0, n_classes):
            y_hat[:, k] = np.sum(w[:, k]*X, axis=1)
        return np.argmax(y_hat, axis=1)
    else:
        return np.sum(w*X, axis=1)

def acuracia(y_hat, y_test):
    """ Retorna a acurácia do modelo"""
    y_test = np.argmax(y_test, axis=1)
    acc = np.sum(y_hat == y_test)/len(y_test)
    print('Acurácia: {:.2f}'.format(acc))

def obter_dataset(input_path, dict_str):
    """ Função lê o dataset e retorna X, y
    
    Arguments:
        input_path {string} -- String com o caminho para o dataset
        dict_str {dict} -- Dicionário com a codificação da variável classe
    Returns:
        (X, y) -- 
    """
    X = np.loadtxt(input_path, delimiter=',', dtype=np.float, usecols=(0,1,2,3))
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)
    y = np.loadtxt(input_path, delimiter=',', dtype=None, usecols=(4), encoding='UTF', converters={4:dict_str.get})
    return X,y

if __name__ == "__main__":

    # Sem normalização
    input_path='./data/iris.data'
    d={"Iris-setosa":[1,0,0], "Iris-versicolor":[0,1,0],"Iris-virginica":[0,0,1]}

    X, y = obter_dataset(input_path, d)
    print(X[0:5, :])
    print(y[0:5])
    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de : ', np.shape(y))

    X_train, y_train, X_test, y_test = utils.divide_dataset(X, y)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    w = regressao_linear(X_train, y_train)
    print('dimensão de w: ', np.shape(w))

    y_hat = preditor_linear(w, X_test)
    acuracia(y_hat, y_test)

    # Normalizando com z_score
    X_z_score = utils.z_score(X, cols=[1, 2, 3, 4])
    X_train, y_train, X_test, y_test = utils.divide_dataset(X_z_score, y)
    w = regressao_linear(X_train, y_train)
    y_hat = preditor_linear(w, X_test)
    acuracia(y_hat, y_test)

    # Normalizando com min max
    X_min_max = utils.min_max(X, cols=[1, 2, 3, 4])
    X_train, y_train, X_test, y_test = utils.divide_dataset(X_min_max, y)
    w = regressao_linear(X_train, y_train)
    y_hat = preditor_linear(w, X_test)
    acuracia(y_hat, y_test)