import numpy as np
import pandas as pd

from .regressao_linear import regressao_linear, preditor_linear, plot_regularizacao
from .regressao_logistica import regressao_logistica, preditor_logistico, regressao_logistica_multiclasse, preditor_logistico_multiclasse

from .svm import SVM
from .twsvm import twsvm, preditor_twsvm
from .kernels import kernel_linear, kernel_polinomial, kernel_rbf

from .RedeNeural import redeNeuralSoftmax, preditorNeuralSoftmax

from .utils import acuracia, Ymulticlasse, CrossValidacaoEstratificada, agrupa_kfolds
from .parametros import obtem_parametros, completa_parametros

def selecionaKernel(kernel_name, c, **parametros):
    """ Seleciona o kernel e retornar o dicionário de hyperparametros"""
    if kernel_name == 'Linear':
        hyper = {'C': c,
                'Kernel': 'Linear'
                }
        return kernel_linear, hyper

    elif kernel_name == 'Polinomial':
        grau = parametros.get('grau')
        assert(grau is not None)
        hyper = {'C': c,
                'Kernel': 'Polinomial',
                'Grau': int(grau)
                }
        return kernel_polinomial, hyper

    elif kernel_name == 'RBF':
        gamma = parametros.get('gamma')
        assert(gamma is not None)
        hyper = {'C': c,
                'Kernel': 'RBF',
                'Gamma': '{:.2f}'.format(gamma),
                }
        return kernel_rbf, hyper

def train_classificador_linear(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df, regularizadores, output_file, save_pickle=False):
    for regularizador in regularizadores:
        print('\nClassificador linear z_score - Regularizador {:.2f}'.format(regularizador))
        w = regressao_linear(X_treino, y_treino, lamdba=regularizador)
        y_hat = preditor_linear(X_teste, w)
        acc = acuracia(y_hat, y_teste, show=False)
        df_run = pd.DataFrame({'Dataset': nome_dataset,
                                'Algoritmo': 'Classificador Linear',
                                'Hyper': str({'Regularizador': regularizador}),
                                'k_fold': k_fold,
                                'Acuracia': acc
                                }, index=[0])
        df = pd.concat([df, df_run])
    if save_pickle:
        df.to_pickle(output_file)
    return df


def train_regressao_logistica(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df, taxas_de_aprendizado, 
                                max_iteracoes, output_file, save_pickle=False):
    # Regressão logística
    multiclasse = len(np.shape(y_treino)) > 1

    for taxa_aprendizado in taxas_de_aprendizado:
        for it_max in max_iteracoes:
            print('\nRegressão logistica z_score - taxa de aprendizagem {:.1f}, máximo de iterações {:.0f}'.format(float(taxa_aprendizado), it_max))
            if multiclasse:
                w_log = regressao_logistica_multiclasse(X_treino, y_treino, taxa_aprendizado=taxa_aprendizado, max_iteracoes=int(it_max))
                y_hat = preditor_logistico_multiclasse(X_teste, w_log)
            else:
                w_log, erros = regressao_logistica(X_treino, y_treino, taxa_aprendizado=taxa_aprendizado, max_iteracoes=int(it_max))
                y_hat = preditor_logistico(X_teste, w_log)

            acc = acuracia(y_hat, y_teste, show=False)
            df_run = pd.DataFrame({'Dataset': nome_dataset,
                                    'Algoritmo': 'Regressão Logística',
                                    'Hyper': str({'Taxa de aprendizado': float(taxa_aprendizado),
                                                  'Max Iter': int(it_max)
                                             }),
                                    'k_fold': k_fold,
                                    'Acuracia': acc
                                    }, index=[0])
            df = pd.concat([df, df_run])
    if save_pickle:
        df.to_pickle(output_file)
    return df

#SVM
def train_SVM_instance(nome_dataset, k_fold, svm_clf, hyper, X_treino, y_treino, X_teste, y_teste, df):
    print('\nSVM - {}'.format(str(hyper)))
    svm_clf.fit(X_treino, y_treino)
    y_hat = svm_clf.predict(X_teste)
    acc = acuracia(y_hat, y_teste, show=False)
    df_run = pd.DataFrame({'Dataset': nome_dataset,
                            'Algoritmo': 'SVM',
                            'Hyper': str(hyper),
                            'k_fold':k_fold,
                            'Acuracia': acc
                            }, index=[0])
    return pd.concat([df, df_run])

def train_SVM(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df, kernels, regularizadores, graus, gammas, output_file, save_pickle=False):
    for kernel_name in kernels:#, kernel_rbf
        for c in regularizadores:
            if kernel_name == 'Linear':
                kernel, hyper = selecionaKernel(kernel_name, c)
                svm_clf = SVM(kernel=kernel, C=c)
                df = train_SVM_instance(nome_dataset, k_fold, svm_clf, hyper, X_treino, y_treino, X_teste, y_teste, df)
            elif kernel_name == 'Polinomial':
                kernel = kernel_polinomial
                for grau in graus:
                    kernel, hyper = selecionaKernel(kernel_name, c, grau=grau)
                    svm_clf = SVM(kernel=kernel, grau=grau, escalar=1, C=c)
                    df = train_SVM_instance(nome_dataset, k_fold, svm_clf, hyper, X_treino, y_treino, X_teste, y_teste, df)
            elif kernel_name == 'RBF':
                kernel = kernel_rbf
                for gamma in gammas:
                    kernel, hyper = selecionaKernel(kernel_name, c, gamma=gamma)
                    svm_clf = SVM(kernel=kernel, gamma=gamma, C=c)
                    df = train_SVM_instance(nome_dataset, k_fold, svm_clf, hyper, X_treino, y_treino, X_teste, y_teste, df)
    if save_pickle:
        df.to_pickle(output_file)
    return df

#TWSVM
def train_TWSVM_instance(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, kernel, parametros, c, hyper, df):
    print('\nTWSVM - {}'.format(str(hyper)))
    z1, z2 = twsvm(X_treino, y_treino, kernel=kernel, parametros=parametros, C_1=c, C_2=c)
    y_hat = preditor_twsvm(X_teste=X_teste, X_treino=X_treino,
                            y_treino=y_treino, kernel=kernel, parametros=parametros, 
                            z1=z1, z2=z2)
    acc = acuracia(y_hat, y_teste, show=True)
    df_run = pd.DataFrame({'Dataset': nome_dataset,
                            'Algoritmo': 'TWSVM',
                            'Hyper': str(hyper),
                            'k_fold': k_fold,
                            'Acuracia': acc
                            }, index=[0])
    return pd.concat([df, df_run])

def train_TWSVM(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df, kernels, regularizadores, graus, gammas, output_file, save_pickle=False):
    for kernel_name in kernels:
        for c in regularizadores:
            if kernel_name == 'Linear':
                kernel, hyper = selecionaKernel(kernel_name, c)
                parametros={}
                df = train_TWSVM_instance(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, kernel, parametros, c, hyper, df)
            elif kernel_name == 'Polinomial':
                for grau in graus:
                    kernel, hyper = selecionaKernel(kernel_name, c, grau=grau)
                    parametros={'Grau':grau}
                    df = train_TWSVM_instance(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, kernel, parametros, c, hyper, df)
            elif kernel_name == 'RBF':
                kernel = kernel_rbf
                for gamma in gammas:
                    kernel, hyper = selecionaKernel(kernel_name, c, gamma=gamma)
                    parametros={'Gamma':gamma}
                    df = train_TWSVM_instance(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, kernel, parametros, c, hyper, df)
    if save_pickle:
        df.to_pickle(output_file)
    return df

#Rede Neural
def train_rede_neural(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df, taxas_de_aprendizado, neuronios, funcoes, 
                        max_iteracoes, output_file, save_pickle=False):
    # caso y seja binário, execute one-hot-encoding
    if len(y_treino.shape) == 1:
        y_treino, y_teste = Ymulticlasse(y_treino, y_teste)
    
    for f_ativacao in funcoes:
        for n_h in neuronios:
            for taxa_aprendizado in taxas_de_aprendizado:
                for max_iter in max_iteracoes:
                    hyper = {'Função de ativação': f_ativacao,
                             'Número de neurônios': int(n_h),
                             'Taxa de aprendizado': '{:.2f}'.format(float(taxa_aprendizado)),
                             'Épocas': int(max_iter)
                             }
                    print('\nRede Neural - {}'.format(hyper))
                    W1, b1, W2, b2, custo = redeNeuralSoftmax(X_treino, y_treino, int(n_h), 
                                                f_ativacao, taxa_aprendizado, max_iter, custo_min=1e-5, plot=False)
                    y_hat = preditorNeuralSoftmax(X_teste, W1, b1, W2, b2, ativacao=f_ativacao)
                    acc = acuracia(y_hat, y_teste)
                    df_run = pd.DataFrame({'Dataset': nome_dataset,
                                            'Algoritmo': 'Rede Neural',
                                            'Hyper': str(hyper),
                                            'k_fold':k_fold,
                                            'Acuracia': acc
                                            }, index=[0])
                    df = pd.concat([df, df_run])
    if save_pickle:
        df.to_pickle(output_file)
    return df

def train_models_run(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, modelos, df_resultados, parametros):

    regularizadores = parametros['Regularizadores']
    taxas_de_aprendizado = parametros['Taxas de aprendizado']
    max_iteracoes = parametros['Max Iteracoes']
    kernels = parametros['Kernels']
    graus = parametros['Pol Graus']
    gammas = parametros['RBF Gammas']
    funcoes = parametros['Funções Ativação']
    neuronios = parametros['Neurônios']

    for modelo in modelos:
        if modelo == 'Classificador Linear':
            output_file = 'resultados/{}_classificador_linear.pickle'.format(nome_dataset)
            df_resultados = train_classificador_linear(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df_resultados, 
                                                        regularizadores, output_file)
        elif modelo == 'Rede Neural':
            output_file='resultados/{}_RedeNeural.pickle'.format(nome_dataset)
            df_resultados = train_rede_neural(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df_resultados, taxas_de_aprendizado, 
                                                neuronios, funcoes, max_iteracoes, output_file)

        elif modelo == 'Regressão Logística':
            output_file='resultados/{}_logistico.pickle'.format(nome_dataset)
            df_resultados = train_regressao_logistica(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df_resultados, 
                                                        taxas_de_aprendizado, max_iteracoes, output_file)
        elif modelo == 'SVM':
            output_file='resultados/{}_SVM.pickle'.format(nome_dataset)
            df_resultados = train_SVM(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df_resultados, kernels, regularizadores,
                                        graus, gammas, output_file)
        elif modelo == 'TWSVM':
            output_file='resultados/{}_TWSVM.pickle'.format(nome_dataset)
            df_resultados = train_TWSVM(nome_dataset, k_fold, X_treino, y_treino, X_teste, y_teste, df_resultados, kernels, regularizadores, 
                                        graus, gammas, output_file)
    
    return df_resultados
    

def treinaCrossValidacao(nome_dataset, X_treino, y_treino, modelos, parametros={}, save_pickles=False):
    """ Função para executar todos os testes dos modelos determinados

    Args:
        nome_dataset (String): Nome do dataset para colocar nos arquivos 
        X_treino (np.array): Atributos de treino
        y_treino (np.array): Classe de treino
        modelos (list): Lista com os modelos a serem executados
        parametros (Dict): Dicionário com os parâmetros a serem testados

    Returns:
        Pandas DataFrame: DataFrame com os resultados
    """

    parametros = obtem_parametros(parametros)

    k_folds = parametros['k_folds']
    k_folds_seed = parametros['k_folds_seed']

    # DataFrame com os resultados
    colunas = ['Dataset', 'Algoritmo', 'Hyper', 'Acuracia']
    df_folds = pd.DataFrame(columns=colunas)

    X_treino_folds = CrossValidacaoEstratificada(X_treino, y_treino, k_folds, k_folds_seed)
    
    for k_fold, fold in enumerate(X_treino_folds):
        # Separando por fold
        outros_ids = set(range(len(X_treino))).difference(set(fold))
        outros_ids = list(outros_ids)

        X_treino_fold = X_treino[fold]
        y_treino_fold = y_treino[fold]
        X_teste_fold = X_treino[outros_ids]
        y_teste_fold = y_treino[outros_ids]

        df_folds = train_models_run(nome_dataset, k_fold, X_treino_fold, y_treino_fold, X_teste_fold, y_teste_fold, modelos, df_folds, parametros)
    
    return df_folds
    
def treinarModelos(nome_dataset, X_treino, y_treino, X_teste, y_teste, modelos, topN=3, parametros={}, save_pickle=True, save_excel=True):
    """ Executa a validação cruzada para verificar quais os melhores hyperparâmetros dividindo o dataset de treino entre treino e validação
        e dos TopN modelos para cada tipo de classificador verifica a sua acurácia no dataset de teste.

    Args:
        nome_dataset: Nome do dataset em que está sendo testado
        X_treino: X treino
        y_treino: Y treino
        X_teste: X teste
        y_teste: Y teste
        modelos ([Lista): Lista dos modelos que serão testados.
        topN (int, optional): Quantos dos melhores conjuntos de hyperparâmetros serão testados por tipo de modelo. Defaults to 3.
        parametros (dict, optional): [description]. Defaults to {}.
        save_pickle (bool, optional): [description]. Defaults to True.
        save_excel (bool, optional): [description]. Defaults to True.

    Returns:
        df_folds: Pandas DataFrame com a performance por fold (testado em X treino)
        df_folds_agrupado: Pandas DataFrame com as estatísticas de cada elemento do set de hyperparâmetros testado
        df_topN: Pandas DataFrame com a perfomance dos topN hyperparametros de cada classificador escolhido (em X teste)
    """
    df_folds = treinaCrossValidacao(nome_dataset, X_treino, y_treino, modelos, parametros=parametros)
    df_folds_agrupado = agrupa_kfolds(df_folds)

    if save_pickle:
        df_folds.to_pickle('resultados/{}_folds.pickle'.format(nome_dataset))
        df_folds_agrupado.to_pickle('resultados/{}_folds_agrupado.pickle'.format(nome_dataset))
    if save_excel:
        df_folds.to_excel('resultados/{}_folds.xls'.format(nome_dataset))
        df_folds_agrupado.to_excel('resultados/{}_folds_agrupado.xls'.format(nome_dataset))

    colunas = ['Dataset', 'Algoritmo', 'Hyper', 'Acuracia']
    df_topN = pd.DataFrame(columns=colunas)

    top_models = df_folds_agrupado.sort_values('mean', ascending=False).groupby('Algoritmo').head(int(topN))
    for _,row in top_models.iterrows():
        modelo = row['Algoritmo']
        parametros = completa_parametros(row['Hyper'])
        df_topN = train_models_run(nome_dataset, None, X_treino, y_treino, X_teste, y_teste,
                                            modelos=[modelo], df_resultados=df_topN, parametros=parametros)
    df_topN = df_topN.merge(df_folds_agrupado, how='left', on=['Dataset', 'Algoritmo', 'Hyper']).drop(columns=['k_fold'])

    if save_pickle:
        df_topN.to_pickle('resultados/{}_topN.pickle'.format(nome_dataset))
    if save_excel:
        df_topN.to_excel('resultados/{}_topN.xls'.format(nome_dataset))

    return df_folds, df_folds_agrupado, df_topN