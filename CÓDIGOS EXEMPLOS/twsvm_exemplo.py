import numpy as np
# Solução do prof. Clodoaldo em Python
X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
K = (X @ X.T + 1)**2
K = np.hstack((K, np.ones((4,1))))
G = K[0:4:3,]
H = K[1:3,:]
eps = 1e-16
Z = G @ np.linalg.pinv(H.T @ H + eps * np.eye(len(H.T @ H))) @ G.T
print(Z)

