import re
import numpy as np
np.random.seed(1234567890)


# generate A
def spddiagm(n):
    mx = np.zeros(shape=(n, n))
    for i in range(n):
        mx[i, i] = 4
    for i in range(n - 1):
        mx[i, i + 1] = -2
        mx[i + 1, i] = -2
    return mx

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

# minimum residual method
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

# conjugate gradient method
def cg(A, b, iter=25):
    residuals = []
    for i in range(2, iter + 2):
        V, T = lanczos(A, b, k=i)
        e = np.zeros(shape=(T.shape[0] - 1, 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T[0:-1, :]
        y = np.linalg.solve(A2, b2)
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
    return np.array(residuals)

def cg2(A, b, iter=25, tol=1e-8):
    x = np.zeros(shape=b.shape)
    r = b.copy()
    rho_1 = 0
    residuals = []
    for i in range(iter):
        z = r
        rho = float(r.T @ z)
        if i > 0:
            beta = rho / rho_1
            p = z + beta * p
        else:
            p = z
        q = A @ p
        alpha = rho / float(p.T @ q)
        x = x + alpha * p
        r = r - alpha * q
        rres = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            rho_1 = rho
    return np.array(residuals)

def cg_tol(A, b, tol=1e-8):
    x = np.zeros(shape=b.shape)
    r = b.copy()
    rho_1 = 0
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        z = r
        rho = float(r.T @ z)
        if cnt > 1:
            beta = rho / rho_1
            p = z + beta * p
        else:
            p = z
        q = A @ p
        alpha = rho / float(p.T @ q)
        x = x + alpha * p
        r = r - alpha * q
        rres = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            rho_1 = rho
    return cnt, np.array(residuals)

# sub question 1
def sub1():
    print()
    A = spddiagm(100)
    b = np.ones(shape=(100, 1))
    r1 = minres(A, b, iter=25)
    r2 = cg(A, b, iter=25)
    r3 = cg2(A, b, iter=25)
    for i in range(25):
        print("| {:<2d} | {:<11.8f} | {:<11.8f} | {:<11.8f} |".format(i + 1, r1[i], r2[i], r3[i]))

# sub question 2
def sub2():
    print()
    A = spddiagm(100)
    b = np.ones(shape=(100, 1))
    cnt, _ = cg_tol(A, b, tol=1e-8)
    print(cnt)

    buffer = []
    for n in (10, 20, 40, 45, 46, 47, 48, 80):
        A = spddiagm(n)
        b = np.ones(shape=(n, 1))
        cnt, _ = cg_tol(A, b, tol=1e-8)
        buffer.append(cnt)
    print(buffer)

# run sub questions
# sub1()
sub2()