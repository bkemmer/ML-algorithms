import numpy as np
from scipy.optimize import minimize

# TWSVM

K_A_CT = np.array([
                [9, 1, 1, 1],
                [1, 9, 1, 1]
                ])
e_1 = np.ones((2,1))
S = np.concatenate((K_A_CT, e_1), axis=1)
eps = 1e-16
TMP_A = (S.T @ S)
TMP1 = TMP_A + eps*np.identity(TMP_A.shape[0])

K_B_CT = np.array([
                [1, 1, 9, 1],
                [1, 1, 1, 9]
                ])
e_2 = np.ones((2,1))
R = np.concatenate((K_B_CT, e_2), axis=1)

# primeiro plano
z_tmp = R @ np.linalg.pinv(TMP1) @ R.T
print(z_tmp)

# Optimization
C_1 = 100
def objective(x):
    x1 = x[0]
    x2 = x[1]
    return -(x1+x2) + 0.039225*np.power((x1+x2),2)


alpha_inicial = np.zeros(2)
print(objective(alpha_inicial))

bound = (0, C_1)
alpha_bounds = (bound, bound)
# const1 = {'type': 'ineq', 'fun': }

sol = minimize(objective, alpha_inicial, method='SLSQP', bounds=alpha_bounds)
print('Soluções alpha:')
alphas = sol.x
print(alphas)

print('Valor em alpha*:')
print(sol.fun)

z1 = -np.linalg.pinv(TMP1) @ R.T @ alphas
print(z1)

w_1 = z1[:-1]
b_1 = z1[-1]

# segundo plano
L = np.concatenate((K_A_CT, e_1), axis=1)
N = np.concatenate((K_B_CT, e_2), axis=1)

eps = 1e-16
TMP_B = (N.T @ N)
TMP2 = TMP_B + eps*np.identity(TMP_B.shape[0])
z_2 = -np.linalg.pinv(TMP2) @ L.T
print(z_2)

z_2_tmp = L @ np.linalg.pinv(TMP2) @ L.T
print(z_2_tmp)

# Optimization
C_2 = 100
def objective(x):
    x1 = x[0]
    x2 = x[1]
    return -(x1+x2) + 0.03925*np.power((x1+x2),2)


gamma_inicial = np.zeros(2)
print(objective(gamma_inicial))

bound = (0, C_2)
gamma_bounds = (bound, bound)

sol = minimize(objective, gamma_inicial, method='SLSQP', bounds=gamma_bounds)
print('Soluções gamma:')
gammas = sol.x
print(gammas)

print('Valor em gamma*:')
print(sol.fun)

z_2 = np.linalg.pinv(TMP2) @ L.T @ gammas
print(z_2)

w_2 = z_2[:-1]
b_2 = z_2[-1]

def dist(k, w, b):
    return np.abs()