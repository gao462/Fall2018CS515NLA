import re
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)


# generate A
def strako(n=30, lambda_1=0.1, lambda_n=100, rho=0.9):
    mx = np.zeros(shape=(n, n))
    for i in range(n):
        d = lambda_1 + i / (n - 1) * (lambda_n - lambda_1) * (rho ** (n - i - 1))
        mx[i, i] = d
    return mx

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = A.shape[0]
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
    return v, alpha, beta

def fillin(v, alpha, beta, n, k=None):
    # allocate matrices
    assert k is None or k > 1
    k = k or len(v)
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

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

# sub question 1
def sub1():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)

    V = lanczos_vecs0[0][0]
    quant0 = [np.linalg.norm(V.T @ V - np.eye(1))]
    V = lanczos_vecs1[0][0]
    quant1 = [np.linalg.norm(V.T @ V - np.eye(1))]
    quant2 = [0]
    for i in range(1, n):
        V, _ = fillin(*lanczos_vecs0, n=n, k=i + 1)
        quant0.append(np.linalg.norm(V.T @ V - np.eye(i + 1)))
        V, _ = fillin(*lanczos_vecs1, n=n, k=i + 1)
        quant1.append(np.linalg.norm(V.T @ V - np.eye(i + 1)))
        quant2.append(0)
    quant0 = np.log10(np.array(quant0) + 1e-20)
    quant1 = np.log10(np.array(quant1) + 1e-20)
    quant2 = np.log10(np.array(quant2) + 1e-20)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(1, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax.plot(np.arange(1, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax.plot(np.arange(1, 31), quant2, color='green', alpha=0.85, linestyle='--', label='SHOULD')
    ax.legend()
    fig.savefig('p2-1.png')

# sub question 2
def sub2():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    quant0 = []
    quant1 = []
    for i in range(0, n):
        v = lanczos_vecs0[0]
        quant0.append(float(v[0].T @ v[i]))
        v = lanczos_vecs1[0]
        quant1.append(float(v[0].T @ v[i]))
    quant0 = np.log10(np.fabs(np.array(quant0)) + 1e-20)
    quant1 = np.log10(np.fabs(np.array(quant1)) + 1e-20)

    ax1.plot(np.arange(1, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax1.plot(np.arange(1, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax1.legend()

    quant0 = []
    quant1 = []
    for i in range(2, n):
        v = lanczos_vecs0[0]
        quant0.append(float(v[i - 2].T @ v[i]))
        v = lanczos_vecs1[0]
        quant1.append(float(v[i - 2].T @ v[i]))
    quant0 = np.log10(np.fabs(np.array(quant0)) + 1e-20)
    quant1 = np.log10(np.fabs(np.array(quant1)) + 1e-20)

    ax2.plot(np.arange(3, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax2.plot(np.arange(3, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax2.legend()

    fig.savefig('p2-2.png')

# sub question 3
def sub3():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)
    print(lanczos_vecs0[2][n])
    print(lanczos_vecs1[2][n])

# sub question 3
def sub4():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=61)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=61)

    quant0 = []
    quant1 = []
    for i in range(1, 61):
        V, T = fillin(*lanczos_vecs0, n=n, k=i + 1)
        quant0.append(np.linalg.norm(A @ V[:, 0:-1] - V @ T))
        V, T = fillin(*lanczos_vecs1, n=n, k=i + 1)
        quant1.append(np.linalg.norm(A @ V[:, 0:-1] - V @ T))
    quant0 = np.log10(np.array(quant0) + 1e-20)
    quant1 = np.log10(np.array(quant1) + 1e-20)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(1, 61), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax.plot(np.arange(1, 61), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax.legend()
    fig.savefig('p2-4.png')

# run sub questions
sub1()
sub2()
sub3()
sub4()