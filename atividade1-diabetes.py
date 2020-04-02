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

def obter_dataset_hepatitis(input_path):
    """ Função lê o dataset e retorna X, y
    
    Arguments:
        input_path {string} -- String com o caminho para o dataset
    Returns:
        (X, y) -- 
    """
    # data = np.genfromtxt(input_path, skip_header=0, delimiter=',', dtype=np.float, names=True)
    data = np.genfromtxt(input_path, skip_header=1, delimiter=',', dtype=np.float)
    X = data[:,:-1]
    # Adicionadno a coluna x_0 que será multiplicada com o viés (bias)
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)
    y = data[:,-1]
    y[y == 0.] = -1
    y[y == 1.] = 1
    return X,y

if __name__ == "__main__":

    # Dataset Diabetes
    # Sem normalização
    input_path='./data/diabetes/diabetes.csv'

    X, y = obter_dataset_hepatitis(input_path)

    print('5 exemplos de X:')
    print(X[0:5, :])
    print('5 exemplos de y:')
    print(y[0:5])
    print('Dimensão de X: ', np.shape(X))
    print('Dimensão de : ', np.shape(y))

    X_train, y_train, X_test, y_test = divide_dataset(X, y)
    print('Dimensão do treino e teste:', np.shape(X_train), np.shape(X_test))

    w = regressao_linear(X_train, y_train)
    print('dimensão de w: ', np.shape(w))
    y_hat = preditor_linear(w, X_test)
    _ = acuracia(y_hat, y_test)

    # Normalizando com z_score
    X_z_score = z_score(X)
    X_train, y_train, X_test, y_test = divide_dataset(X_z_score, y)
    w = regressao_linear(X_train, y_train)
    y_hat = preditor_linear(w, X_test)
    _ = acuracia(y_hat, y_test)

    # Normalizando com min max
    X_min_max = min_max(X)
    X_train, y_train, X_test, y_test = divide_dataset(X_min_max, y)
    w = regressao_linear(X_train, y_train)
    y_hat = preditor_linear(w, X_test)
    _ = acuracia(y_hat, y_test)

    # Utilizando a regressao logistica
    print("Regressão logística:")
    w_log = regressao_logistica(X_train, y_train, taxa_aprendizado=0.5, max_iteracoes=1000)

    y_hat_log = preditor_logistico(X_test, w_log)
    _ = acuracia(y_hat_log, y_test)


    # # # Com regularização
    # # Variando 0<=lambda<1 
    # plot_regularizacao(X_train, y_train, X_test, y_test,
    #                     output_file_name="./imgs/diabetes_acuracia_regressor_linear.png")
    # # Variando 0<=lambda<10
    # plot_regularizacao(X_train, y_train, X_test, y_test, 
    #                     limits_min=0, limits_max=1000, 
    #                     output_file_name="./imgs/diabetes_acuracia_regressor_linear10.png")

    # # Variando 0<=lambda<100
    # plot_regularizacao(X_train, y_train, X_test, y_test, 
    #                     limits_min=0, limits_max=10000, 
    #                     output_file_name="./imgs/diabetes_acuracia_regressor_linear100.png")
