import numpy as np
import matplotlib.pyplot as plt

from utils import acuracia, divide_dataset, z_score, min_max, plot_erros
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao
from regressao_logistica import regressao_logistica_multiclasse, preditor_logistico_multiclasse
from redeNeural import redeNeuralSoftmax

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

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

    X_z_score_train, X_z_score_test = z_score(X_train, X_test)

    feature_set = X_z_score_train
    one_hot_labels = y_train

    instances = feature_set.shape[0]
    attributes = feature_set.shape[1]
    output_labels = 3

    wo = np.random.rand(attributes,output_labels)
    # bo = np.random.randn(output_labels)
    lr = 10e-4

    error_cost = []
    N = X.shape[0]
    X = np.concatenate([X, np.ones((N, 1))], axis=1)
    J = []
    for epoch in range(5000):
    ############# feedforward

        # Phase 2
        zo = np.dot(feature_set, wo) #+ bo
        ao = softmax(zo)

    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = ao - one_hot_labels
        dzo_dwo = feature_set

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo

    ########## Phases 2

        # dzo_dah = wo
        # dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        # dah_dzh = sigmoid_der(zh)
        # dzh_dwh = feature_set
        # dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        # dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        # wh -= lr * dcost_wh
        # bh -= lr * dcost_bh.sum(axis=0)

        wo -= lr * dcost_wo
        # bo -= lr * dcost_bo.sum(axis=0)

        if epoch % 200 == 0:
            loss = np.sum(-one_hot_labels * np.log(ao))
            J.append(loss)
            print('Loss function value: ', loss)
            error_cost.append(loss)
    plt.plot(J)
    plt.show()