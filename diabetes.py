"""
## Dataset: Pima Indians Diabetes Database
[fonte](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### Descrição:
O dataset consiste em diferentes variáveis preditoras médicas (independentes) e uma variável classe (dependente).
As variávies independentes incluem: número de gravidezes que o paciente teve, seu BMI, nível de insulina, idade entre outros.

### Atributos:

1. Pregnancies - Gravidezes - número de vezes em que a pessoa já engravidou
2. Glucose - Nível de glicose - concentração de de glicose no plasma em 2 horas em um teste de tolerância oral.
3. BloodPressure - Diastolic blood pressure (mm Hg) - Pressão sanguínea diastólica
4. SkinThickness - Triceps skin fold thickness (mm) - Grossura da pele via o quanto é possível dobrar do triceps
7. Insulin - Nível de insulina 2-Hour serum insulin (mu U/ml) - 
8. BMI - Body mass index - Índice de massa corporal
9. Age - Idade
10. Outcome - Variável classe (0 ou 1) - 268 de 768 exemplos são da classe 1, os outros são 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modelos.utils import divide_dataset, z_score, agrupa_kfolds
from modelos.train import treinarModelos

from modelos.parametros import inicializa_parametros_para_teste


# def obter_dataset_diabetes(input_path):
#     """ Função lê o dataset e retorna X, y
    
#     Arguments:
#         input_path {string} -- String com o caminho para o dataset
#     Returns:
#         (X, y) -- 
#     """
#     # data = np.genfromtxt(input_path, skip_header=0, delimiter=',', dtype=np.float, names=True)
#     data = np.genfromtxt(input_path, skip_header=1, delimiter=',', dtype=np.float)
#     X = data[:,:-1]
#     # Adicionadno a coluna x_0 que será multiplicada com o viés (bias)
#     X = np.concatenate((np.ones((len(X),1)), X), axis=1)
#     y = data[:,-1]
#     y[y == 0.] = -1
#     y[y == 1.] = 1
#     return X,y

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
   
    modelos = ['Classificador Linear', 'Regressão Logística', 'SVM', 'TWSVM', 'Rede Neural']
    nome_dataset = 'diabetes'

    #usando os hyperparâmetros defaults
    parametros={}
    # parametros = inicializa_parametros_para_teste()
    df_folds, df_folds_agrupado, df_topN = treinarModelos(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, 
                                                            topN=3, parametros=parametros, save_pickle=True, save_excel=True)
if __name__ == "__main__":
    main()
    
