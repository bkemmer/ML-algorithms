import numpy as np
import matplotlib.pyplot as plt

def acuracia(y_hat, y_test, show=True):
    """ Retorna a acurácia do modelo"""
    # Problema de classificação binária
    if len(np.shape(y_test)) < 2:
        acc = np.sum(np.sign(y_hat) == y_test)/len(y_test)
    # Caso tenha mais classes
    else:
        y_test = np.argmax(y_test, axis=1)
        acc = np.sum(y_hat == y_test)/len(y_test)
    if show:
        print('Acurácia: {:.4f}'.format(acc))
    return acc

def divide_dataset(X, y, fator=0.7, seed=42):
    """Função para dividir o dataset entre o fator para treino e 1 - fator para teste.

    Arguments:
        X {matriz} -- atributos
        y {matriz} -- classes

    Keyword Arguments:
        fator {float} -- fator {float} -- percentual para treino e (1-fator) para teste (default: {0.7})
        seed {int} -- seed para fixar a divisão dos dados (default: {42})

    Returns:
         (X_train, y_train, X_test, y_test) -- tupla com exemplos de treino e teste
    """

    N = len(X)
    n_train = int(fator * N)

    np.random.seed(seed)
    indices = np.random.permutation(N)
    training_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train = X[training_idx]
    y_train = y[training_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return (X_train, y_train, X_test, y_test)

def z_score(X_train, X_test, cols=None):
    """ Aplica a normalização ~N(0,1) também conhecida como z_score
    
    Arguments:
        X_train {Matriz} -- Matriz de atributos de treinamento
        X_test {Matriz} -- Matriz de atributos de teste
        cols {list} -- colunas a ser aplicada a normalização
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos de treino transformada
        Matriz -- Matriz de atributos dos exemplos de teste transformada
    """
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    if cols is None:
        cols = np.arange(0,X_train.shape[1], dtype=int)
    for col in cols:
        mean_train = np.mean(X_train[:,col])
        std_train = np.std(X_train[:,col])
        # normalizando os exemplos de treinamento
        X_train[:,col] = (X_train[:,col] - mean_train)/std_train
        # normalizando os exemplos de teste
        X_test[:,col] = (X_test[:,col] - mean_train)/std_train
    return X_train, X_test
    
def min_max(X_train, X_test, cols=None, slack=0.5):
    """ Aplica a normalização min max
    
    Arguments:
        X_train {Matriz} -- Matriz de atributos de treinamento
        X_test {Matriz} -- Matriz de atributos de teste
        cols {list} -- colunas a ser aplicada a normalização
        slack {float} -- multiplicador para o valor de máximo e mínimo (1+slack)
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos de treino transformada
        Matriz -- Matriz de atributos dos exemplos de teste transformada
    """ 
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    if cols is None:
        cols = np.arange(0,X_train.shape[1], dtype=int)
    for col in cols:
        min_train = np.min(X_train[:,col])*(1 + slack)
        max_train = np.max(X_train[:,col])*(1 + slack)
        # normalizando os exemplos de treinamento
        X_train[:,col] = (X_train[:,col] - min_train)/(max_train - min_train)
        # normalizando os exemplos de teste
        X_test[:,col] = (X_test[:,col] - min_train)/(max_train - min_train)
    return X_train, X_test
    
def completar_com(X_train, X_test, func, cols=None):
    """ Aplica a funcção "func" para todos os missing values
        somente da coluna específicada em "cols" ou em todas 
    
    Arguments:
        X_train {Matriz} -- Matriz de atributos de treinamento
        X_test {Matriz} -- Matriz de atributos de teste
        func {function} -- função de agregação para ser aplicada nos dados faltantes
    
    Keyword Arguments:
        cols {list} -- Lista de colunas a aplicar (default: {None} - Todas)
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos de treino transformada
        Matriz -- Matriz de atributos dos exemplos de teste transformada
    """
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    if cols is None:
        cols = np.arange(0,X_train.shape[1], dtype=int)
    for col in cols:
        val = func(X_train[~np.isnan(X_train[:,col]), col])
        X_train[np.isnan(X_train[:,col]), col] = val
        X_test[np.isnan(X_test[:,col]), col] = val
    return X_train, X_test

def plot_erros(erros, output_fname, figsize=(15,15), title=''):
    """ Função para plotar os erros

    Arguments:
        erros {Array} -- Vetor com os erros a cada época
        output_fname {string} -- path para salvar a imagem
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(erros)
    ax1.set_xlabel("Iterações")
    ax1.set_ylabel("J(theta): Custo")
    plt.title(str(title) + ' Função de erro a cada iteração')
    fig.savefig(output_fname)
    plt.show()
    
def Ymulticlasse(y_train, y_test):
    """ Função recebe dois vetores binários e transforma eles em one-hot-encoding

    Arguments:
        y_train {np.array} -- vetor de classes de treino
        y_test {np.array} -- vetor de classes de treino

    Returns:
        (np.array, np.array) -- vetor de classes de treino e teste
    """
    classes = sorted(set(y_train).union(set(y_test)))
    y_train_multi = np.zeros((len(y_train), len(classes)))
    y_test_multi = np.zeros((len(y_test), len(classes)))
    for i, classe in enumerate(classes):
        y_train_multi[:, i] = np.where(y_train == classe, 1, 0)
        y_test_multi[:, i] = np.where(y_test == classe, 1, 0)
    return y_train_multi, y_test_multi