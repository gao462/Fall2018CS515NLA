import re
import numpy as np
np.random.seed(1234567890)


# load csv files
def read_csv(path):
    file = open(path)
    content = file.readlines()
    file.close()
    data = []
    for line in content:
        items = line.strip().split(',')
        items = [float(itr) for itr in items]
        data.append(items)
    return np.array(data)
candyland_cells = read_csv('candyland-cells.csv')
candyland_coords = read_csv('candyland-coords.csv')
candyland_matrix = read_csv('candyland-matrix.csv')

# create transition matrix from list of indices and values
# $T_{i, j}$: Probability from $j$ to $i$
def matrix_from_liv(shape, liv):
    mx = np.zeros(shape)
    for i, j, val in liv:
        mx[int(i) - 1][int(j) - 1] = val
    return mx
trans_mx = matrix_from_liv((140, 140), candyland_matrix)

# set A, b, and solve x = inv(A) @ b
A = np.eye(140) - trans_mx.T
b = np.ones((140, 1))
b[133, 0] = 0
A2 = A.T @ A
b2 = A.T @ b

# neumann
def neumann_tol(A, b, tol=1e-8):
    M = np.eye(A.shape[0]) - A
    r = b.copy()
    x = b.copy()
    cnt = 0
    while cnt < 10000:
        cnt += 1
        r = M @ r
        x = x + r
        if np.linalg.norm(b - A @ x) / np.linalg.norm(b) < tol:
            break
        else:
            pass
    return x, cnt

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
    return cnt, residuals

# conjugate gradient method
def cg_tol(A, b, tol=1e-8):
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        V, T = lanczos(A, b, k=cnt + 1)
        e = np.zeros(shape=(T.shape[0] - 1, 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T[0:-1, :]
        y = np.linalg.solve(A2, b2)
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            pass
    return cnt, residuals

_, cnt = neumann_tol(A, b, tol=1e-8)
print(cnt)
cnt, residuals = minres_tol(A2, b2, tol=1e-8)
print(cnt)
cnt, residuals = cg_tol(A2, b2, tol=1e-8)
print(cnt)