import numpy as np
from scipy.linalg import ldl
np.random.seed(1234567890)

def cholesky_rank1_update(L0, D0, u):
    n = L0.shape[0]
    D1 = D0.copy()
    L1 = L0.copy()
    a = 1
    w = u
    for j in range(n):
        p = w[j, 0]
        D1[j, j] = D0[j, j] + a * p * p
        b = p * a / D1[j, j]
        a = D0[j, j] * a / D1[j, j]
        for r in range(j + 1, n):
            w[r, 0] = w[r, 0] - p * L0[r, j]
            L1[r, j] = L0[r, j] + b * w[r, 0]
    return L1, D1

N = 5
A = np.random.normal(size=(N, N))
A = A @ A.T
u = np.random.normal(size=(N, 1))
L, D, _ = ldl(A)
L1, D1, _ = ldl(A + u @ u.T)
L2, D2 = cholesky_rank1_update(L, D, u)
print(np.linalg.norm(L1 - L2))
print(np.linalg.norm(D1 - D2))
