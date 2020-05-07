import numpy as np
from numpy import random

def grad(x1,x2):
    return np.array([x1-x2, 2*x2 -x1 - 1])

def r_x(x):
        return np.array([x[0]-x[1], -x[0]+2*x[1]-1])

Hessiana = np.array([ [1,0],[-1,1] ]) @ np.array([[1,-1],[0,1]])

#Inicializando
alpha = 1
k = 0
interaction = 500
# x = np.array([  random.randint(0, 10)  ,  random.randint(0, 10)  ])
x = [1, 4]
print(x)
#Metodo de LM
mu = 0.00001
grad_r = np.array([[1, -1], [0, 1]])

for k in range(0,interaction):
    mu = 0.00001
    # x = x - alpha*np.linalg.inv(Hessiana + 0.00001) @ grad(x[0],x[1])   
    alpha = bissecao(x, d, gradF_X=grad_r.T @ r_x)
    x = x - alpha*np.linalg.inv(grad_r.T @ grad_r + mu) @ grad_r.T @ r_x(x)
     
    print(x, k)
    if np.linalg.norm(grad(x[0],x[1])) < 1.0e-09:
        break
x
