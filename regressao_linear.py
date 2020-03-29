# Regressão linear

import numpy as np

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
    # TODO: arrumar para multiclasse
    return np.sign(np.sum(w*X, axis=1))

def funcao_erro(y_hat, y):
    """ Função de erro"""
    return np.sign(y_hat - y)


def obter_dataset(input_path='./data/iris.data', 
                  dict_str=d
                 ):
    X = np.loadtxt(input_path, delimiter=',', dtype=np.float, usecols=(0,1,2,3))
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)
    y = np.loadtxt(input_path, delimiter=',', dtype=None, usecols=(4), encoding='UTF', converters={4:dict_str.get})
    return X, y


input_path='./data/iris.data'
d={"Iris-setosa":[1,-1,-1], "Iris-versicolor":[-1,1,-1],"Iris-virginica":[-1,-1,1]}
X, y = obter_dataset(input_path, d)
print(X[0:5, :])
print(y[0:5])

print('Dimensão de X: ', np.shape(X))
print('Dimensão de : ', np.shape(y))

w = regressao_linear(X, y)
print('dimensão de w: ', np.shape(w))

# FIXME:
y_hat = np.sign(np.sum(w[:,0]*X, axis=1))
np.sum((y_hat > 0)) 