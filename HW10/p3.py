import re
import numpy as np
np.random.seed(1234567890)


# generate A
def gen_A(n):
    e = np.ones(shape=(n, 1))
    return np.eye(n) + e @ e.T

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = b.shape[0]
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # allocate buffer
    v = [None for i in range(k)]
    alpha = [None for i in range(k - 1)]
    beta = [None for i in range(k)]

    # initial loop is special
    beta[0] = np.linalg.norm(b)
    v[0] = b / beta[0]
    alpha[0] = float(v[0].T @ A @ v[0])
    w = A @ v[0] - alpha[0] * v[0]
    beta[1] = np.linalg.norm(w)
    v[1] = w / beta[1]

    # loop for the remaining part
    for i in range(1, k - 1):
        alpha[i] = float(v[i].T @ A @ v[i])
        w = A @ v[i] - alpha[i] * v[i] - beta[i] * v[i - 1]
        beta[i + 1] = np.linalg.norm(w)
        v[i + 1] = w / beta[i + 1]

    # fill in values
    for i in range(k):
        V_mx[:, [i]] = v[i]
    for i in range(k - 1):
        if i > 0:
            T_mx[i - 1, i] = beta[i]
        else:
            pass
        T_mx[i, i] = alpha[i]
        T_mx[i + 1, i] = beta[i + 1]
    return V_mx, T_mx

# minimum residual method# minimum residual method
def minres(A, b, iter=25):
    residuals = []
    for i in range(2, iter + 2):
        V, T = lanczos(A, b, k=i)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
    return np.array(residuals)

def minres_tol(A, b, tol=1e-8):
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        V, T = lanczos(A, b, k=cnt + 1)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            pass
    return cnt, np.array(residuals)

n = 200
A = gen_A(n)
b = np.random.normal(size=(n, 1))
x = b - b.sum() / (n + 1)
cnt, _ = minres_tol(A, b)
print(cnt)