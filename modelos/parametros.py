def obtem_parametros(parametros):

    parametros['k_folds'] = parametros.get('K-folds', 5)
    parametros['k_folds_seed'] = parametros.get('K-folds seed', 42)

    parametros['Regularizadores'] = parametros.get('Regularizadores', [0, .5, 1, 2, 5, 1e1, 1e2, 1e3, 1e4, 1e5])
    parametros['Taxas de aprendizado'] = parametros.get('Taxas de aprendizado', [0.1, .5, 1, 5, 10, 100])
    parametros['Max Iteracoes'] = parametros.get('Max Iteracoes', [1e2, 1e3, 1e4])
    parametros['Kernels'] = parametros.get('Kernels', ['Linear', 'Polinomial']) #, kernel_rbf
    parametros['Pol Graus'] = parametros.get('Pol Graus', [2,3,4])
    parametros['RBF Gammas'] = parametros.get('RBF Gammas', [0.1, 0.3, 0.5, 0.75, 1])
    parametros['Funções Ativação'] = parametros.get('Funções Ativação', ['sigmoid', 'relu'])
    parametros['Neurônios'] = parametros.get('Neurônios', [1, 1e1, 1e2, 1e3])

    return parametros

def hyper2parametros(hyper, chaves):
    for chave in chaves:
        if hyper.get(chave, None) is not None:
            return [hyper.get(chave, None)]
    return []

def completa_parametros(hyper):
    hyper = eval(hyper)
    # print(hyper)
    parametros = {}
    parametros['Regularizadores'] = hyper2parametros(hyper, ['Regularizador', 'C'])
    parametros['Taxas de aprendizado'] = hyper2parametros(hyper, ['Taxa de aprendizado'])
    parametros['Taxas de aprendizado'] = [float(x) for x in parametros['Taxas de aprendizado']]

    parametros['Kernels'] = hyper2parametros(hyper, ['Kernel'])
    parametros['Pol Graus'] = hyper2parametros(hyper, ['Grau'])
    #FIXME: Confirmar se está funcionando o Gamma
    parametros['RBF Gammas'] = hyper2parametros(hyper, ['Gamma'])
    parametros['Funções Ativação'] = hyper2parametros(hyper, ['Função de ativação'])
    parametros['Neurônios'] = hyper2parametros(hyper, ['Número de neurônios'])
    parametros['Max Iteracoes'] = hyper2parametros(hyper, ['Épocas', 'Max Iter'])
    # print(parametros)
    # print()
    return parametros


def inicializa_parametros_para_teste():
    parametros = {}
    parametros['Regularizadores'] = [1]
    parametros['Taxas de aprendizado'] = [1]
    parametros['Max Iteracoes'] = [1e2]
    parametros['Kernels'] = ['Linear', 'Polinomial'] #, 'RBF'
    parametros['Pol Graus'] = [2]
    parametros['RBF Gammas'] = [0.5]
    parametros['Funções Ativação'] = ['sigmoid', 'relu']
    parametros['Neurônios'] = [100]
    
    return parametros
