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

from utils import acuracia, divide_dataset, z_score, min_max, completar_com
from regressao_linear import regressao_linear, preditor_linear, plot_regularizacao

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

if __name__ == "__main__":

    # Dataset Hepatitis
    # Sem normalização
    input_path='./data/hepatitis/hepatitis.data'

    X, y = obter_dataset_hepatitis(input_path)
    # Completando todos os dados faltantes com a média de sua respectiva coluna
    X = completar_com(X, np.mean)
    
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

    # # Com regularização
    # Variando 0<=lambda<1 
    plot_regularizacao(X_train, y_train, X_test, y_test,
                        output_file_name="./imgs/hepatitis_acuracia_regressor_linear.png")
    # Variando 0<=lambda<10
    plot_regularizacao(X_train, y_train, X_test, y_test, 
                        limits_min=0, limits_max=1000, 
                        output_file_name="./imgs/hepatitis_acuracia_regressor_linear10.png")

    # Variando 0<=lambda<100
    plot_regularizacao(X_train, y_train, X_test, y_test, 
                        limits_min=0, limits_max=10000, 
                        output_file_name="./imgs/hepatitis_acuracia_regressor_linear100.png")
