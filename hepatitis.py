"""
# Atividade 1 - Análises no dataset: Hepatitis

## Dataset: Hepatitis Data Set

[fonte](https://archive.ics.uci.edu/ml/datasets/Hepatitis)

- Número de instâncias: 155
- Tem valores faltantes: Sim
- Número de atributos: 19

## Atributos

1. Class: DIE, LIVE
2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
3. SEX: male, female
4. STEROID: no, yes
5. ANTIVIRALS: no, yes
6. FATIGUE: no, yes
7. MALAISE: no, yes
8. ANOREXIA: no, yes
9. LIVER BIG: no, yes
10. LIVER FIRM: no, yes
11. SPLEEN PALPABLE: no, yes
12. SPIDERS: no, yes
13. ASCITES: no, yes
14. VARICES: no, yes
15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
17. SGOT: 13, 100, 200, 300, 400, 500,
18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
20. HISTOLOGY: no, yes
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import acuracia, divide_dataset, z_score, min_max, completar_com, plot_erros, Ymulticlasse
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao

from regressao_logistica import regressao_logistica, preditor_logistico

#SVM
from twsvm import twsvm, preditor_twsvm, kernel_pol
from svm import SVM, kernel_linear, kernel_polinomial, kernel_rbf
from sklearn.svm import SVC

from RedeNeural import redeNeuralSoftmax, preditorNeuralSoftmax

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
    # Sem normalização
    input_path='./data/hepatitis/hepatitis.data'

    X, y = obter_dataset_hepatitis(input_path)
    # print('5 exemplos de X:')
    # print(X[0:5, :])
    # print('5 exemplos de y:')
    # print(y[0:5])
    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de y: ', np.shape(y))

    # Completando todos os dados faltantes com a média de sua respectiva coluna
    X_train, y_train, X_test, y_test = divide_dataset(X, y)
    X_train, X_test = completar_com(X_train, X_test, np.mean)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))
    w = regressao_linear(X_train, y_train)
    print('dimensão de w: ', np.shape(w))
    y_hat = preditor_linear(X_test, w)
    print('\nRegressão linear s/normalização:')
    _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # w = regressao_linear(X_z_score_train, y_train)
    # y_hat = preditor_linear(X_z_score_test, w)
    # print('\nRegressão linear z_score:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com min max
    # X_min_max_train, X_min_max_test = min_max(X_train, X_test, slack=2)
    # w = regressao_linear(X_train, y_train)
    # y_hat = preditor_linear(X_min_max_test, w)
    # print('\nRegressão linear min_max:')
    # _ = acuracia(y_hat, y_test)

    # # # Utilizando a regressao logistica
    # # title = "Regressão logística:"
    # # print(title)
    # # taxa_aprendizado = 0.1
    # # w_log, erros = regressao_logistica(X_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/hepatitis_erro_logistica_{}.png'.format(taxa_aprendizado), figsize=(10,5))
    # # y_hat_log = preditor_logistico(X_test, w_log)
    # # print('\nRegressão logistica z_score:')
    # # _ = acuracia(y_hat_log, y_test)


    # # Normalizando com z_score a regressão logística
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # taxa_aprendizado = 0.1
    # w_log, erros = regressao_logistica(X_z_score_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=2000)
    # # plot_erros(erros, output_fname='./imgs/hepatitis_erro_logistica_{}_zscore.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico(X_z_score_test, w_log)
    # print('\nRegressão logistica z_score:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score a regressão logística
    # X_minmax_train, X_minmax_test = min_max(X_train, X_test)
    # taxa_aprendizado = 0.1
    # w_log, erros = regressao_logistica(X_minmax_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=2000)
    # # plot_erros(erros, output_fname='./imgs/hepatitis_erro_logistica_{}_minmax.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico(X_minmax_test, w_log)
    # print('\nRegressão logistica min_max:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score TW-SVM
    # c=1.0
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # z_1, z_2 = twsvm(X_z_score_train, y_train, C_1=c, C_2=c)
    # y_hat = preditor_twsvm(X_z_score_test, X_z_score_train, y_train, kernel_pol, z_1, z_2) #, y_test
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_svm_{}_zscore.png'.format(), figsize=(10,5))
    # print('\nTW-SVM z_score:')
    # _ = acuracia(y_hat, y_test, show=True)

    # # Normalizando com z_score SVM
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # c=1
    # svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1, C=c)
    # svm_clf.fit(X_z_score_train, y_train)
    # y_hat = svm_clf.predict(X_z_score_test)
    # print('\nSVM z_score:')
    # _ = acuracia(y_hat, y_test, show=True)

    # # Normalizando com z_score SVM - SCIKIT
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # clf = SVC(kernel='poly', degree=2, coef0=1, C=2)
    # clf.fit(X_z_score_train, y_train)
    # y_scikit = clf.predict(X_z_score_test)
    # print('\nSVM Scikit-learn z_score:')
    # _ = acuracia(y_scikit, y_test)

    # Normalizando com z_score a Rede Neural com Softmax
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    y_train_multi, y_test_multi = Ymulticlasse(y_train, y_test)
    taxa_aprendizado = 0.5
    W1, b1, W2, b2, custo = redeNeuralSoftmax(X_z_score_train, y_train_multi, h_size=100, ativacao='sigmoid', taxa_aprendizado=1, max_iteracoes=10000, custo_min=1e-5, plot=True)
    y_hat = preditorNeuralSoftmax(X_z_score_test, W1, b1, W2, b2, ativacao='sigmoid')
    print('\nRede Neural Softmax z_score:')
    _ = acuracia(y_hat, y_test_multi)    
    
if __name__ == "__main__":
    main()
