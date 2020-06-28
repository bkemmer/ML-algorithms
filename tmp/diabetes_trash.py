    # df_resultados.to_pickle('resultados/df_resultados_reg_lin.pickle')
    # # # Normalizando com min max
    # # X_min_max_train, X_min_max_test = min_max(X_train, X_test)
    # # w = regressao_linear(X_train, y_train)
    # # y_hat = preditor_linear(X_min_max_test, w)
    # # print('\nRegressão linear min_max:')



        # w = regressao_linear(X_train, y_train)
    # print('dimensão de w: ', np.shape(w))
    # y_hat = preditor_linear(X_test, w)
    # print('\nRegressão linear s/normalização:')
    # _ = acuracia(y_hat, y_test)

        # Dataset Diabetes
    # Sem normalização
    # input_path='./data/diabetes/diabetes.csv'
    # X, y = obter_dataset_hepatitis(input_path)


        # # Normalizando com min_max a regressão logística
    # X_minmax_train, X_minmax_test = min_max(X_train, X_test)
    # taxa_aprendizado = 0.5
    # w_log, erros = regressao_logistica(X_minmax_train, y_train, taxa_aprendizado=taxa_aprendizado, max_iteracoes=1000)
    # # plot_erros(erros, output_fname='./imgs/diabetes_erro_logistica_{}_minmax.png'.format(taxa_aprendizado), figsize=(10,5))
    # y_hat = preditor_logistico(X_minmax_test, w_log)
    # print('\nRegressão logistica min_max:')
    # _ = acuracia(y_hat, y_test)


    # def kernel_pol(X, C, pol=2, escalar=1):
#     """ Aplica o kernel polinomial na matriz X

#     Arguments:
#         X {Matriz} -- Matriz a ser aplicado o kernel
#         C {Matriz} -- Matriz concatenada entre as duas classes

#     Keyword Arguments:
#         pol {int} -- Grau do polinômio (default: {2})
#         escalar {int} -- escalar adicionado (default: {1})

#     Returns:
#         Matriz -- Matriz de kernel
#     """
#     return np.power((X @ C.T) + escalar, pol)
