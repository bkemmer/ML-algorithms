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

from utils import acuracia, divide_dataset, z_score, min_max
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao

from regressao_logistica import regressao_logistica, preditor_logistico

from sklearn.linear_model import LogisticRegression

from sklearn import svm as svm_scikit

from RedeNeuralSoftmax import redeNeuralSoftmax, preditorNeuralSoftmax

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

    return X_train,y_train, X_test, y_test

if __name__ == "__main__":

    # Dataset Diabetes
    # Sem normalização
    # input_path='./data/diabetes/diabetes.csv'
    # X, y = obter_dataset_hepatitis(input_path)

    input_path_train ='./data/diabetes/dataset_train.txt'
    input_path_test ='./data/diabetes/dataset_teste.txt'

    # X_train, y_train, X_test, y_test = divide_dataset(X, y)
    X_train, y_train, X_test, y_test = obter_dataset_diabetesv2(input_path_train, input_path_test)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    # # Normalizando com z_score
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # w = regressao_linear(X_z_score_train, y_train)
    # y_hat = preditor_linear(w, X_z_score_test)
    # print('Regressão linear z_score:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score a regressão logística
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # w_log, erros = regressao_logistica(X_z_score_train, y_train, taxa_aprendizado=0.5, max_iteracoes=1000)
    # plt.plot(erros)
    # plt.show()
    # y_hat = preditor_logistico(X_z_score_test, w_log)
    # print('Regressão logistica z_score:')
    # _ = acuracia(y_hat, y_test)

    # clf = LogisticRegression(random_state=42).fit(X_z_score_train, y_train)
    # y_scikit = clf.predict(X_z_score_test)

    # _ = acuracia(y_scikit, y_test)
    
    # # Normalizando com z_score
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    clf = svm_scikit.SVC(C=1, kernel='poly', degree=2)
    clf.fit(X_z_score_train, y_train)
    y_scikit = clf.predict(X_z_score_test)
    _ = acuracia(y_scikit, y_test)
    

    svm_clf = svm_scikit.SVC(kernel='poly', degree=2, coef0=1, shrinking=False)
    svm_clf.fit(X_z_score_train, y_train)
    y_hat = svm_clf.predict(X_z_score_test)
    print('\nSVM scikit z_score:')
    _ = acuracia(y_hat, y_test)