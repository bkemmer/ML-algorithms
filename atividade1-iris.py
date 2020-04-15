"""
# Atividade 1 - Análises no dataset: Iris
## Dataset:  Iris Data Set 

[fonte](https://archive.ics.uci.edu/ml/datasets/Iris?spm=a2c4e.11153940.blogcont603256.5.333b1d6f05ZggC)

Atributos:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
 - Iris Setosa
 - Iris Versicolour
 - Iris Virginica
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import acuracia, divide_dataset, z_score, min_max
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao

def obter_dataset_iris(input_path, dict_str):
    """ Função lê o dataset e retorna X, y
    
    Arguments:
        input_path {string} -- String com o caminho para o dataset
        dict_str {dict} -- Dicionário com a codificação da variável classe
    Returns:
        (X, y) -- 
    """
    X = np.loadtxt(input_path, delimiter=',', dtype=np.float, usecols=(0,1,2,3))
    y = np.loadtxt(input_path, delimiter=',', dtype=None, usecols=(4), encoding='UTF', converters={4:dict_str.get})
    return X,y

if __name__ == "__main__":

    # Dataset Iris
    # Sem normalização
    input_path='./data/iris/iris.data'
    d={"Iris-setosa":[1,0,0], "Iris-versicolor":[0,1,0],"Iris-virginica":[0,0,1]}

    X, y = obter_dataset_iris(input_path, d)
    print(X[0:5, :])
    print(y[0:5])
    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de : ', np.shape(y))

    X_train, y_train, X_test, y_test = divide_dataset(X, y)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    w = regressao_linear(X_train, y_train)
    print('dimensão de w: ', np.shape(w))

    y_hat = preditor_linear(X_test, w)
    _ = acuracia(y_hat, y_test)

    # Normalizando com z_score
    X_z_score_train, X_z_score_test = z_score(X_train, X_test, cols=[0, 1, 2, 3])
    w = regressao_linear(X_z_score_train, y_train)
    y_hat = preditor_linear(X_z_score_test, w)
    _ = acuracia(y_hat, y_test)

    # Normalizando com min max
    X_min_max_train, X_min_max_test = min_max(X_train, X_test, cols=[0, 1, 2, 3])
    w = regressao_linear(X_min_max_train, y_train)
    y_hat = preditor_linear(X_min_max_test, w)
    _ = acuracia(y_hat, y_test)

    # Com regularização
    # Variando 0<=lambda<1 
    plot_regularizacao(X_train, y_train, X_test, y_test,
                        output_file_name="./imgs/iris_acuracia_regressor_linear.png")
    # Variando 0<=lambda<10
    plot_regularizacao(X_train, y_train, X_test, y_test, 
                        limits_min=0, limits_max=1000, 
                        output_file_name="./imgs/iris_acuracia_regressor_linear10.png")

    # Variando 0<=lambda<100
    plot_regularizacao(X_train, y_train, X_test, y_test, 
                        limits_min=0, limits_max=10000, 
                        output_file_name="./imgs/iris_acuracia_regressor_linear100.png")
