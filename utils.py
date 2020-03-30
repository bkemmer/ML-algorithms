import numpy as np

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

def z_score(X, cols):
    """ Aplica a normalização ~N(0,1) também conhecida como z_score
    
    Arguments:
        X {Matriz} -- Matriz de atributos dos exemplos
        cols {list} -- colunas a ser aplicada a normalização
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos transformada
    """
    for col in cols:
        X[:,col] = (X[:,col] - np.mean(X[:,col]))/np.std(X[:,col])
    return X
    
def min_max(X, cols):
    """ Aplica a normalização min max
    
    Arguments:
        X {Matriz} -- Matriz de atributos dos exemplos
        cols {list} -- colunas a ser aplicada a normalização
    
    Returns:
        Matriz -- Matriz de atributos dos exemplos transformada
    """  
    for col in cols:
        X[:,col] = (X[:,col] - np.min(X[:,col]))/(np.max(X[:,col]) - np.min(X[:,col]))
    return X
    