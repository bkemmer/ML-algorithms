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
import pandas as pd

from modelos.utils import z_score, divide_dataset, agrupa_kfolds
from modelos.train import treinarModelos

from modelos.parametros import inicializa_parametros_para_teste

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

def main():
    # Dataset Iris
    # Sem normalização
    input_path='./data/iris/iris.data'
    d={"Iris-setosa":[1,0,0], "Iris-versicolor":[0,1,0],"Iris-virginica":[0,0,1]}

    X, y = obter_dataset_iris(input_path, d)
    # print(X[0:5, :])
    # print(y[0:5])
    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de : ', np.shape(y))

    X_treino, y_treino, X_teste, y_teste = divide_dataset(X, y)
    print('Dimensão do treino e teste:', np.shape(X_treino), np.shape(X_teste))
    
    # Normalizando com z_score
    X_treino_zscore, X_teste_zscore = z_score(X_treino, X_teste)
    
    modelos = ['Classificador Linear', 'Regressão Logística', 'Rede Neural']
    nome_dataset = 'iris'

    #usando os hyperparâmetros defaults
    parametros={}
    # parametros = inicializa_parametros_para_teste()
    df_folds, df_folds_agrupado, df_topN = treinarModelos(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, 
                                                            topN=3, parametros=parametros, save_pickle=True, save_excel=True)

if __name__ == "__main__":
    main()
