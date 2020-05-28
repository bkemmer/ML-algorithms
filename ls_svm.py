# Atividade 4: Bruno Kemmer 
# Sin 5016

import numpy as np


# LS-SVM

C = 10
A = np.array([
    [9+(1/C), -1, -1, 1, -1],
    [-1, 9+(1/C), 1, -1, 1],
    [-1, 1, 9+(1/C), -1, 1],
    [1, -1, -1, 9+(1/C), -1],
    [-1, 1, 1, -1, 0]
])
b = np.array([1, 1, 1, 1, 0])

x = np.linalg.solve(A, b)

