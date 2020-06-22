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

from utils import acuracia, divide_dataset, z_score, min_max, plot_erros, Ymulticlasse
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao
from regressao_logistica import regressao_logistica_multiclasse, preditor_logistico_multiclasse
# from RedeNeuralSoftmax import redeNeuralSoftmax, preditorNeuralSoftmax

#SVM
from twsvm import twsvm, preditor_twsvm, kernel_pol
from svm import SVM, kernel_linear, kernel_polinomial, kernel_rbf
from sklearn.svm import SVC

from svm_internet import SVM as SVM_int
from svm_internet import linear_kernel, polynomial_kernel, gaussian_kernel

from RedeNeural import redeNeuralSoftmax, preditorNeuralSoftmax

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

    X_train, y_train, X_test, y_test = divide_dataset(X, y)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    w = regressao_linear(X_train, y_train)
    print('dimensão de w: ', np.shape(w))

    # y_hat = preditor_linear(X_test, w)
    # print('\nRegressão linear s/ normalização:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test, cols=[0, 1, 2, 3])
    # w = regressao_linear(X_z_score_train, y_train)
    # y_hat = preditor_linear(X_z_score_test, w)
    # print('\nRegressão linear z_score:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com min max
    # X_min_max_train, X_min_max_test = min_max(X_train, X_test, cols=[0, 1, 2, 3])
    # w = regressao_linear(X_min_max_train, y_train)
    # y_hat = preditor_linear(X_min_max_test, w)
    # print('\nRegressão linear min_max:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score a regressão logística
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # taxa_aprendizado = 0.1
    # w_log = regressao_logistica_multiclasse(X_z_score_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}_zscore.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico_multiclasse(X_z_score_test, w_log)
    # print('\nRegressão logistica z_score:')
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score a regressão logística
    # print('\nRegressão logistica min_max:')
    # X_minmax_train, X_minmax_test = min_max(X_train, X_test)
    # taxa_aprendizado = 0.1
    # w_log = regressao_logistica_multiclasse(X_minmax_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}_minmax.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico_multiclasse(X_minmax_test, w_log)
    # _ = acuracia(y_hat, y_test)

    # # Normalizando com z_score SVM
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # c=1
    # models = {}
    # for i in range(0, y_train.shape[1]):
    #     models[i] = SVM(kernel=kernel_polinomial, grau=2, escalar=1, C=c)
    #     models[i].fit(X_z_score_train, y_train[:,i])
    
    # results = []
    # for key,model in models.items():
    #     results.append(model.predict(X_z_score_test))

    # y_hat = svm_clf.predict(X_z_score_test)
    # print('\nSVM z_score:')
    # _ = acuracia(y_hat, y_test, show=True)

    # Normalizando com z_score SVM
    X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    c=2
    for i in range(0, y_train.shape[1]):
        svm_clf = SVM(kernel=kernel_polinomial, grau=2, escalar=1, C=c)
        svm_clf.fit(X_z_score_train, y_train[:,i])
        y_hat = svm_clf.predict(X_z_score_test)
        print('\nSVM z_score:{}'.format(i))
        _ = acuracia(y_hat, y_test[:,i])

    # c=1.0
    # for i in range(0, y_train.shape[1]):
    #     z_1, z_2 = twsvm(X_z_score_train, y_train[:,i], C_1=c, C_2=c)
    #     y_hat = preditor_twsvm(X_z_score_test, X_z_score_train, y_train[:,i], kernel_pol, z_1, z_2) #, y_test
    #     print('\nTW-SVM z_score:')
    #     _ = acuracia(y_hat, y_test[:,i])

    for i in range(0, y_train.shape[1]):
        print("\n{}".format(i))
        # Normalizando com z_score SVM - SCIKIT
        X_z_score_train, X_z_score_test = z_score(X_train, X_test)
        clf = SVC(kernel='poly', degree=2, coef0=1, C=2)
        clf.fit(X_z_score_train, y_train[:,i])
        y_scikit = clf.predict(X_z_score_test)
        print('\nSVM Scikit-learn z_score:{}'.format(i))
        _ = acuracia(y_scikit, y_test[:,i])
        # print("intercept: {:.4f}".format(clf.intercept_[0]))
        # print("sv:")
        # print(clf.support_vectors_)
        # print("sv_idx:")
        # print(clf.support_)
    
    # for i in range(0, y_train.shape[1]):
    #     # Normalizando com z_score SVM - SCIKIT
    #     clf_int = SVM_int(kernel=polynomial_kernel, C=2)
    #     clf_int.fit(X_z_score_train, y_train[:,i])
    #     y_int = clf_int.predict(X_z_score_test)
    #     print('\nSVM internet z_score:{}'.format(i))
    #     _ = acuracia(y_int, y_test[:,i])

    # # Normalizando com z_score TW-SVM
    # c=1.0
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)    
    # # Ymulticlasse()
    # z_1, z_2 = twsvm(X_z_score_train, y_train, C_1=c, C_2=c)
    # y_hat = preditor_twsvm(X_z_score_test, X_z_score_train, y_train, kernel_pol, z_1, z_2) #, y_test
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_svm_{}_zscore.png'.format(), figsize=(10,5))
    # print('\nTW-SVM z_score:')
    # _ = acuracia(y_hat, y_test, show=True)




    # # Normalizando com z_score SVM - SCIKIT
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # clf = SVC(kernel='poly', degree=2, coef0=1, C=2)
    # clf.fit(X_z_score_train, y_train)
    # y_scikit = clf.predict(X_z_score_test)
    # print('\nSVM Scikit-learn z_score:')
    # _ = acuracia(y_scikit, y_test)

    # # Normalizando com z_score a softmax
    # print('\nRede Neural Softmax z_score:')
    # X_z_score_train, X_z_score_test = z_score(X_train, X_test)
    # taxa_aprendizado = 0.1
    # w_soft, erros = redeNeuralSoftmax(X_z_score_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=5000, plot=False)
    # plt.plot(erros)
    # plt.show()
    # # plot_erros(erros, output_fname='./imgs/iris_erro_redeSoftMax_{}_zscore.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditorNeuralSoftmax(X_z_score_test, w_soft)
    # _ = acuracia(y_hat, y_test)

if __name__ == "__main__":
    main()
