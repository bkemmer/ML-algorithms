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
import pandas as pd

from modelos.utils import divide_dataset, completar_com, z_score
from modelos.train import treinarModelos

from modelos.parametros import inicializa_parametros_para_teste

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
    
    modelos = ['Classificador Linear', 'Regressão Logística', 'SVM', 'TWSVM', 'Rede Neural']
    nome_dataset = 'hepatitis'

    #usando os hyperparâmetros defaults
    parametros={}
    # parametros = inicializa_parametros_para_teste()
    df_folds, df_folds_agrupado, df_topN = treinarModelos(nome_dataset, X_treino_zscore, y_treino, X_teste_zscore, y_teste, modelos, 
                                                            topN=3, parametros=parametros, save_pickle=True, save_excel=True)

if __name__ == "__main__":
    main()
