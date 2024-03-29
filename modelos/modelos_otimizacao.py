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
    return np.power(x[0],2) + 2*np.power(x[1],2) - 2*x[0]*x[1] - 2*x[1] 

def gradF_X(x):
    """ Retorna o gradiente da função f_x
    """
    return np.array([2*x[0]-2*x[1], 4*x[1]-2*x[0]-2])

def hessiana_X(x):
    """ Retorna a matriz Hessiana da função f_x
    """
    return np.array([[2, -2],[-2, 4]])

def grad_descendente(x, taxa_aprendizado_fixa=None, iter_max=1000, max_error=1e-3):


    J = []
    normas = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        d = -gradF_X(x)

        norm = np.abs(np.linalg.norm(d))
        normas.append(norm)
        if  norm < max_error:
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J, normas, i

        if taxa_aprendizado_fixa is None:
            alpha = bissecao(x, d, gradF_X=gradF_X)
        else:
            alpha = taxa_aprendizado_fixa
        x += alpha*d

        f_val = f_X(x)
        # if i % 2 == 0:
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
    return x, J, normas, i

def grad_descendente_newton(x, taxa_aprendizado_fixa=None, iter_max=1000, max_error=1e-3):

    J = []
    normas = []
    f_val = f_X(x)
    J.append(f_val)
    print ("Valor inicial de F_X: %f" %(f_val))

    for i in range(iter_max):

        grad_x = gradF_X(x)
      
        norm = np.abs(np.linalg.norm(grad_x))
        normas.append(norm)
        if  norm < max_error:         
            print ("Valor de F_X na iteração final %i: %f" %(i, f_val))
            return x, J, normas, i

        H_x = hessiana_X(x)
        
        d = -np.linalg.inv(H_x) @ grad_x
        
        if taxa_aprendizado_fixa is None:
            alpha = bissecao(x, d, gradF_X=gradF_X)
        else:
            alpha = taxa_aprendizado_fixa
        x += alpha*d

        f_val = f_X(x)
        J.append(f_val)
        print ("Valor de F_X na iteração %i: %f" %(i, f_val))
  
    return x, J, normas, i

def bissecao(x, direcao, gradF_X, alpha_superior=None, h_derivada_min=1e-3, max_error=1e-3):

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
        if np.abs(np.linalg.norm(h_derivada)) < h_derivada_min:
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

def gradiente_conjugado(x0, n, taxa_aprendizado_fixa=None, metodo=1, iter_max=1000, max_error=1e-3):
    """
        Utilizando a aproximação Fletcher-Reeves (FR)
    """
    J = []
    d = -gradF_X(x0)
    x = x0
    for i in range(iter_max):
        if taxa_aprendizado_fixa is None:
            alpha = bissecao(x, d, gradF_X=gradF_X)
        else:
            alpha = taxa_aprendizado_fixa
        x_proximo = x + alpha*d
        fx = f_X(x_proximo)
        
        J.append(fx)

        g = -gradF_X(x)
        norm = np.abs(np.linalg.norm(g))
        normas.append(norm)
        if  norm < max_error:
            return x_proximo, J, normas, i

        g_proximo = -gradF_X(x_proximo)
        if i % n != 0:
            if metodo == 1:
                #será utilizado o método Fletcher-Reeves
                b = (g_proximo.T @ g_proximo) / (g.T @ g)
            else:
                # caso contrário será utilizado o método Polak-Ribiére
                b = (g_proximo.T @ (g_proximo - g)) / (g.T @ g)

            d = g + b*d
        else:
            d = g
        x = x_proximo
    return x, J, normas, i


def Levenberf_Marquardt(x, mu=0.0001, iter_max=1000, max_error=1e-3):
    def r_x(x):
        return np.array([x[0]-x[1], -x[0]+2*x[1]-1])
    grad_r = np.array([[1, -1], [0, 1]])
    
    J = []
    normas = []

    for i in range(iter_max):

        a = np.linalg.inv(grad_r.T @ grad_r + mu)
        d = grad_r.T @ r_x(x)
        x -= a @ d

        norm = np.abs(np.linalg.norm(d))
        normas.append(norm)
        J.append(f_X(x))
        if norm < max_error:
            return x, J, normas, i
    
    return x, J, normas, i



x = [1, 4]
alpha=0.1
show = False

x, J, normas, i = grad_descendente(x, taxa_aprendizado_fixa=alpha)
print("Gradiente com taxa de aprendizado fixa:")
print("X={}".format(x))
print("normas[-1]={:.2f} na iteração: {:d}\n".format(normas[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Descendente \ncom taxa de aprendizado fixa')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_passo_fixo.png')
if show: 
    plt.show()
plt.clf()


x = [1, 4]

x, J, normas, i = grad_descendente(x)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Descendente utilizando bisseção')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_passo_variavel.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]

x, J, normas, i = grad_descendente_newton(x, taxa_aprendizado_fixa=alpha)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Descendente utilizando \nmétodo de newton (taxa de aprendizado fixa)')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_newton.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]

x, J, normas, i = grad_descendente_newton(x)
print("Gradiente com Passo Variável:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Descendente \nutilizando método de newton (bisseção)')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_descendente_newton_bissecao.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]

x, J, normas, i = gradiente_conjugado(x, 2, taxa_aprendizado_fixa=alpha)
print("Gradiente Conjugado com taxa de aprendizado fixa:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Conjugado taxa de aprendizado fixa')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_conjugado_fixo.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]

x, J, normas, i = gradiente_conjugado(x, 2)
print("Gradiente Conjugado meodo bisseção:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Conjugado \n(bisseção) - Fletcher-Reeves')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_conjugado_bissecao_FR.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]
x, J, normas, i = gradiente_conjugado(x, 2, metodo=2)
print("Gradiente Conjugado Bisseção:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Gradiente Conjugado \n(bisseção) - Polak-Ribiére')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./grad_conjugado_bissecao_PR.png')
if show: 
    plt.show()
plt.clf()

x = [1, 4]
x, J, normas, i = Levenberf_Marquardt(x)
print("Levenberf Marquardt:")
print("X={}".format(x))
print("J[-1]={:.2f} na iteração: {:d}".format(J[-1],i))
plt.plot(J)
plt.axhline(y=-1, color='black', linestyle='dashed', alpha=0.3)
plt.title('Convergência de F(X) - Levenberf Marquardt')
plt.xlabel('Iterações')
plt.ylabel('F(X)')
plt.savefig('./Levenberf_Marquardt.png')
if show: 
    plt.show()
plt.clf()