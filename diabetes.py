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

from utils import acuracia, divide_dataset, z_score, min_max, plot_erros, Ymulticlasse
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao

from regressao_logistica import regressao_logistica, preditor_logistico

from RedeNeuralSoftmax import redeNeuralSoftmax, preditorNeuralSoftmax

from twsvm import twsvm, preditor_twsvm, kernel_pol
from svm import SVM, kernel_linear, kernel_polinomial, kernel_rbf
# TODO: deletar
from svm_internet import SVM as SVM_int
from svm_internet import linear_kernel, polynomial_kernel, gaussian_kernel
from sklearn.svm import SVC 

import pandas as pd

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
    # Dataset Diabetes
    # Sem normalização
    # input_path='./data/diabetes/diabetes.csv'
    # X, y = obter_dataset_hepatitis(input_path)

    input_path_train ='./data/diabetes/dataset_train.txt'
    input_path_test ='./data/diabetes/dataset_teste.txt'

    # X_train, y_train, X_test, y_test = divide_dataset(X, y)
    X_train, y_train, X_test, y_test = obter_dataset_diabetesv2(input_path_train, input_path_test)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    # w = regressao_linear(X_train, y_train)
    # print('dimensão de w: ', np.shape(w))
    # y_hat = preditor_linear(X_test, w)
    # print('\nRegressão linear s/normalização:')
    # _ = acuracia(y_hat, y_test)

    # Normalizando com z_score
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    w = regressao_linear(X_z_score_train, y_train)
    y_hat = preditor_linear(X_z_score_test, w)
    print('\nRegressão linear z_score:')
    _ = acuracia(y_hat, y_test)

    # # Normalizando com min max
    # X_min_max_train, X_min_max_test = min_max(X_train, X_test)
    # w = regressao_linear(X_train, y_train)
    # y_hat = preditor_linear(X_min_max_test, w)
    # print('\nRegressão linear min_max:')
    # _ = acuracia(y_hat, y_test)

    # # # Utilizando a regressao logistica
    # # title = "Regressão logística:"
    # # print(title)
    # # taxa_aprendizado = 0.5
    # # w_log, erros = regressao_logistica(X_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}.png'.format(taxa_aprendizado), figsize=(10,5))
    # # y_hat_log = preditor_logistico(X_test, w_log)
    # # _ = acuracia(y_hat_log, y_test)


    # Normalizando com z_score a regressão logística
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    taxa_aprendizado = 0.5
    w_log, erros = regressao_logistica(X_z_score_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}_zscore.png'.format(taxa_aprendizado), figsize=(10,5))
    y_hat = preditor_logistico(X_z_score_test, w_log)
    print('\nRegressão logistica z_score:')
    _ = acuracia(y_hat, y_test)

    # # Normalizando com min_max a regressão logística
    # X_minmax_train, X_minmax_test = min_max(X_train, X_test)
    # taxa_aprendizado = 0.5
    # w_log, erros = regressao_logistica(X_minmax_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}_minmax.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico(X_minmax_test, w_log)
    # print('\nRegressão logistica min_max:')
    # _ = acuracia(y_hat, y_test)

    # Normalizando com z_score TW-SVM
    c=2.0
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    z_1, z_2 = twsvm(X_z_score_train, y_train, C_1=c, C_2=c)
    y_hat = preditor_twsvm(X_z_score_test, X_z_score_train, y_train, kernel_pol, z_1, z_2) #, y_test
    # plot_erros(erros, output_fname='./imgs/diabetes_erro_svm_{}_zscore.png'.format(), figsize=(10,5))
    print('\nTW-SVM z_score:')
    _ = acuracia(y_hat, y_test, show=True)

    # Normalizando com z_score SVM
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    c=2
    svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1, C=c)
    svm_clf.fit(X_z_score_train, y_train)
    y_hat = svm_clf.predict(X_z_score_test)
    print('\nSVM z_score:')
    _ = acuracia(y_hat, y_test, show=True)

    # Normalizando com z_score SVM - SCIKIT
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    clf = SVC(kernel='poly', degree=2, coef0=1, C=2)
    clf.fit(X_z_score_train, y_train)
    y_scikit = clf.predict(X_z_score_test)
    print('\nSVM Scikit-learn z_score:')
    _ = acuracia(y_scikit, y_test)

    # Normalizando com z_score SVM - SCIKIT
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    clf = SVM_int(kernel=polynomial_kernel, C=2)
    clf.fit(X_z_score_train, y_train)
    y_int = clf.predict(X_z_score_test)
    print('\nSVM internet z_score:')
    _ = acuracia(y_int, y_test)


    # # Normalizando com z_score a Rede Neural com Softmax
    # print('\nRede Neural Softmax z_score:')
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # y_train_multi, y_test_multi = Ymulticlasse(y_train, y_test)
    # taxa_aprendizado = 0.5
    # w_soft, erros = redeNeuralSoftmax(X_z_score_train, y_train_multi, taxa_aprendizado=taxa_aprendizado, max_iteracoes=5000, plot=False)
    # plot_erros(erros, output_fname='./imgs/diabetes_erro_redeSoftMax_{}_zscore.png'.format(taxa_aprendizado), figsize=(10,5), title='Rede Neural Softmax')
    # y_hat = preditorNeuralSoftmax(X_z_score_test, w_soft)
    # _ = acuracia(y_hat, y_test_multi)

if __name__ == "__main__":
    main()
    
