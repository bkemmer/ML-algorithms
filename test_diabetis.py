import pandas as pd 
import numpy as np

from modelos.utils import acuracia, divide_dataset, z_score, min_max, plot_erros, Ymulticlasse, completar_com, CrossValidacaoEstratificada
from modelos.train import train_models

from modelos.parametros import inicializa_parametros_para_teste

# # df = pd.read_pickle('resultados/df_resultados_SVM.pickle')
# df = pd.read_pickle('resultados/diabetes.pickle')
# print(df)
# df.to_excel('resultados/diabetes.xls')
def obter_dataset_diabetesv2(input_path_train, input_path_test):
    """ Função lê o dataset e retorna X, y
    
    Arguments:
        input_path_train {string} -- String com o caminho para o dataset dividido de treino
        input_path_test {string} -- String com o caminho para o dataset dividido de teste
    Returns:
        (X, y) -- 
    """
    data_train = np.genfromtxt(input_path_train, skip_header=1, delimiter='\t', dtype=np.float)
    X_train = data_train[:,:-1]
    y_train = data_train[:,-1]
    y_train[y_train == 0.] = -1
    y_train[y_train == 1.] = 1

    data_test = np.genfromtxt(input_path_test, skip_header=1, delimiter='\t', dtype=np.float)
    X_test = data_test[:,:-1]
    y_test = data_test[:,-1]
    y_test[y_test == 0.] = -1
    y_test[y_test == 1.] = 1

    return X_train, y_train, X_test, y_test

def main():

    input_path_train ='./data/diabetes/dataset_train.txt'
    input_path_test ='./data/diabetes/dataset_teste.txt'

    # X_train, y_train, X_test, y_test = divide_dataset(X, y)
    X_treino, y_treino, X_teste, y_teste = obter_dataset_diabetesv2(input_path_train, input_path_test)
    print('Dimensão do treino e teste:', np.shape(X_treino), np.shape(X_teste))

    # Normalizando com z_score
    X_treino_zscore, X_teste_zscore = z_score(X_treino, X_teste)

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
