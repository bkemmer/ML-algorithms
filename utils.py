import numpy as np

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
    """[summary]

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

def z_score(X, cols=None):
    """ Aplica a normalização ~N(0,1) também conhecida como z_score
    
    Arguments:
        X {Matriz} -- Matriz de atributos dos exemplos
        cols {list} -- colunas a ser aplicada a normalização
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos transformada
    """
    X_ = np.copy(X)
    if cols is None:
        cols = np.arange(1,X_.shape[1])
    for col in cols:
        X_[:,col] = (X_[:,col] - np.mean(X_[:,col]))/np.std(X_[:,col])
    return X_
    
def min_max(X, cols=None):
    """ Aplica a normalização min max
    
    Arguments:
        X {Matriz} -- Matriz de atributos dos exemplos
        cols {list} -- colunas a ser aplicada a normalização
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos transformada
    """ 
    X_ = np.copy(X)
    if cols is None:
        cols = np.arange(1,X_.shape[1])
    for col in cols:
        X_[:,col] = (X_[:,col] - np.min(X_[:,col]))/(np.max(X_[:,col]) - np.min(X_[:,col]))
    return X_
    
def completar_com(X, func, cols=None):
    """ Aplica a funcção "func" para todos os missing values
        somente da coluna específicada em "cols" ou em todas 
    
    Arguments:
        X {ndarray} -- X
        func {function} -- função de agregação para ser aplicada nos dados faltantes
    
    Keyword Arguments:
        cols {list} -- Lista de colunas a aplicar (default: {None} - Todas)
    
    Returns:
        np.ndarray -- X modificado
    """
    X_ = np.copy(X)
    if cols is None:
        cols = np.arange(1,X_.shape[1])
    for col in cols:
        val = func(X_[~np.isnan(X_[:,col]), col])
        X_[np.isnan(X_[:,col]), col] = val

    return X_

