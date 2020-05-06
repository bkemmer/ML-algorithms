''' Atividade 2 - Bruno Kemmer - 5910474
Considere o problema abaixo
Min f(x1,x2) = x12 + 2x22+-2x1x2-2x2
O objetivo desta atividade é resolver o problema acima, usando os algoritmos de otimização
irrestrita visto em aula. O aluno deve entregar os códigos implementados e um relatório
comparando a convergência dos diferentes algoritmos.
'''
import numpy as np
import matplotlib.pyplot as plt

def f_X(x):
    """ f(x1, x2) = x1^2 + 2*x2^2 -2X1X2-2X2
    """
    return np.power(x[0],2) + 2*np.power(x[1],2) - 2*x[0]*x[1] - x[1] 

def gradF_X(x):
    """ Retorna o gradiente da função f_x
    """
    return np.array([2*x[0]-2*x[1], 4*x[1]-2*x[0]-2])

def hessiana_X(x):
    """ Retorna a matriz Hessiana da função f_x
    """
    return np.array([[2, -2],[-2, 4]])

def grad_descendente_passo_fixo(x, learning_rate=0.1, iter_max=1000, max_error=1e-2):

    J = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        grad_x = gradF_X(x)
        x -= learning_rate*grad_x

        f_val = f_X(x)
        # if i % 2 == 0:
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
  
        if np.abs(f_val) < max_error:
            J.append(f_val)
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J
    return x, J


def grad_descendente_bissecao(x, iter_max=1000, max_error=1e-2):

    J = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        grad_x = gradF_X(x)
        passo = bissecao(x, -grad_x)
        x -= passo*grad_x

        f_val = f_X(x)
        # if i % 2 == 0:
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
  
        if np.abs(f_val) < max_error:
            J.append(f_val)
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J
    return x, J


def grad_descendente_newton_passo_fixo(x, learning_rate=0.1, iter_max=1000, max_error=1e-2):

    J = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        grad_x = gradF_X(x)
        H_x = hessiana_X(x)
        
        d = -np.linalg.inv(H_x) @ grad_x
        # passo = bissecao(x, d)
        x += learning_rate*d

        f_val = f_X(x)
        # if i % 2 == 0:
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
  
        if np.abs(f_val) < max_error:
            J.append(f_val)
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J
    return x, J

def grad_descendente_newton_bissecao(x, iter_max=1000, max_error=1e-2):

    J = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        grad_x = gradF_X(x)
        H_x = hessiana_X(x)
        
        d = -np.linalg.inv(H_x) @ grad_x
        passo = bissecao(x, d)
        x += passo*d

        f_val = f_X(x)
        # if i % 2 == 0:
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
  
        if np.abs(f_val) < max_error:
            J.append(f_val)
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J
    return x, J

def bissecao(x, direcao, alpha_superior=None, h_derivada_min=1e-3, max_error=1e-3):

    alpha_inferior = 0
    if alpha_superior is None:
        alpha_superior = np.random.rand(1)[0]
    
    xn = x + alpha_superior*direcao
    grad_x = gradF_X(xn)
    h_derivada = grad_x.T @ direcao
    while(h_derivada) < 0:
        alpha_superior *= 2
        xn = x + alpha_superior*direcao
        grad_x = gradF_X(xn)
        h_derivada = grad_x.T @ direcao
    
    iter_maximas = np.ceil(np.log((alpha_superior - alpha_inferior)/max_error))
    iter_maximas = int(np.ceil(iter_maximas))
    # print("Número de iterações máximas encontrada: {}".format(iter_maximas))

    for i in range(iter_maximas + 1):
        alpha_medio = (alpha_inferior + alpha_superior)/2
        if np.abs(h_derivada) < h_derivada_min:
            return alpha_medio
        xn = x + alpha_medio*direcao
        grad_x = gradF_X(xn)
        h_derivada = grad_x.T @ direcao
        if h_derivada > 0:
            alpha_superior = alpha_medio
        elif h_derivada < 0:
            alpha_inferior = alpha_medio
        else:
            return alpha_medio
    return alpha_medio



x = [1, 4]

x, J = grad_descendente_passo_fixo(x)
print("Gradiente com Passo Fixo:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}\n".format(J[-1], len(J)))
plt.plot(J)
plt.title('Convergência de F(X) - Gradiente Descendente com passo fixo')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_passo_fixo.png')
plt.show()


x = [1, 4]

x, J = grad_descendente_bissecao(x)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1], len(J)))
plt.plot(J)
plt.title('Convergência de F(X) - Gradiente Descendente utilizando bisseção')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_passo_variavel.png')
plt.show()

x = [1, 4]

x, J = grad_descendente_newton_passo_fixo(x)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1], len(J)))
plt.plot(J)
plt.title('Convergência de F(X) - Gradiente Descendente utilizando método de newton (passo fixo)')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_newton.png')
plt.show()

x, J = grad_descendente_newton_bissecao(x)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1], len(J)))
plt.plot(J)
plt.title('Convergência de F(X) - Gradiente Descendente utilizando método de newton (bisseção)')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_newton_bissecao.png')
plt.show()

#parei no 2h33min aula3