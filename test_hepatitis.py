import pandas as pd 
import numpy as np

from modelos.utils import acuracia, divide_dataset, z_score, min_max, plot_erros, Ymulticlasse, completar_com, CrossValidacaoEstratificada
from modelos.train import train_models

from modelos.parametros import inicializa_parametros_para_teste

# # df = pd.read_pickle('resultados/df_resultados_SVM.pickle')
# df = pd.read_pickle('resultados/diabetes.pickle')
# print(df)
# df.to_excel('resultados/diabetes.xls')

def obter_dataset_hepatitis(input_path):
    """ Função lê o dataset e retorna X, y
    
    Arguments:
        input_path {string} -- String com o caminho para o dataset
    Returns:
        (X, y) -- 
    """
    data = np.genfromtxt(input_path, delimiter=',', dtype=np.float, missing_values='?')
    X = data[:,1:]
    y = data[:,0]
    y[y == 1.] = -1
    y[y == 2.] = 1
    return X,y

def main():
    # Dataset Hepatitis
    input_path='./data/hepatitis/hepatitis.data'

    X, y = obter_dataset_hepatitis(input_path)

    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de y: ', np.shape(y))

    # Completando todos os dados faltantes com a média de sua respectiva coluna
    X_treino, y_treino, X_teste, y_teste = divide_dataset(X, y)
    X_treino, X_teste = completar_com(X_treino, X_teste, np.mean)
    print('Dimensão do treino e teste:', np.shape(X_treino), np.shape(X_teste))
    
    # Normalizando com z_score
    X_treino_zscore, X_teste_zscore = z_score(X_treino, X_teste)
   
    # y_pos = np.sum(y_treino) / len(y_treino)
    # print('Proporção nos testes: {:.2f}'.format(y_pos))

    X_treino_folds = CrossValidacaoEstratificada(X_treino, y_treino, folds=4)
    for i, fold in enumerate(X_treino_folds):
        others = [x for x in range(len(X_treino_folds)) if x != i]
        for other in others:
            [print(x) for x in fold if x in X_treino_folds[other]]

    modelos = ['ClassificadorLinear', 'RegressaoLogistica', 'SVM', 'TWSVM', 'RedeNeural']
    nome_dataset = 'diabetes'
    parametros = inicializa_parametros_para_teste()

    df_resultados = train_models(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, parametros)
    df_resultados.to_pickle('resultados/testes/{}_testes.pickle'.format(nome_dataset))
    df_resultados.to_excel('resultados/testes/{}_testes.xls'.format(nome_dataset))
   
if __name__ == "__main__":
    main()