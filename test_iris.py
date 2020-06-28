import pandas as pd 
import numpy as np

from modelos.utils import acuracia, divide_dataset, z_score, min_max, plot_erros, Ymulticlasse, completar_com, CrossValidacaoEstratificada
from modelos.train import train_models

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
   

    X_treino_folds = CrossValidacaoEstratificada(X_treino, y_treino, folds=4)
    for i, fold in enumerate(X_treino_folds):
        others = [x for x in range(len(X_treino_folds)) if x != i]
        for other in others:
            [print(x) for x in fold if x in X_treino_folds[other]]

    modelos = ['ClassificadorLinear', 'RegressaoLogistica', 'RedeNeural']
    nome_dataset = 'diabetes'
    parametros = inicializa_parametros_para_teste()

    df_resultados = train_models(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, parametros)
    df_resultados.to_pickle('resultados/testes/{}_testes.pickle'.format(nome_dataset))
    df_resultados.to_excel('resultados/testes/{}_testes.xls'.format(nome_dataset))
   
if __name__ == "__main__":
    main()

