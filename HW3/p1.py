import numpy as np

# laplacian construction from homework 2
def laplacian(N, f):
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    fvec = np.zeros(((N + 1) ** 2, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            fvec[idx_mx[i, j] - 1, 0] = f(i / N, j / N)
            if i == 0 or i == N or j == 0 or j == N:
                liv.append([idx_mx[i, j], idx_mx[i, j], -4])
            else:
                for (dx_i, dx_j, val) in ops:
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    return A, fvec

# compute relative residual
def rres(A, b, x_head):
    return np.linalg.norm(A @ x_head - b) / np.linalg.norm(b)

# compute the largest absolute eigenvalue
def rho(A):
    return np.fabs(np.linalg.eig(A)[0]).max()


# sub question 1
def sub1():
    print()
    A, f = laplacian(10, lambda x, y: 1)
    A = -A
    f = -f
    print((np.linalg.eig(A)[0] >= 0).all())
    print(rho(A))

# sub question 2
def sub2():
    print()
    A, f = laplacian(20, lambda x, y: 1)
    A = -A
    f = -f
    print((np.linalg.eig(A)[0] >= 0).all())
    print(rho(A))

# sub question 3
def sub3():
    print()
    A_10, f_10 = laplacian(10, lambda x, y: 1)
    A_20, f_20 = laplacian(20, lambda x, y: 1)
    A_10 = -A_10
    f_10 = -f_10
    A_20 = -A_20
    f_20 = -f_20
    space = np.linspace(0.25 * 0.999, 2 / max(rho(A_10), rho(A_20)), num=20, endpoint=True)
    for alpha in space:
        u_10 = f_10.copy()
        u_20 = f_20.copy()
        cnt = 0
        while (rres(A_10, f_10, u_10) > 1e-5 or \
               rres(A_20, f_20, u_20) > 1e-5) and \
              cnt < 5000:
            u_10 = u_10 + alpha * (f_10 - A_10 @ u_10)
            u_20 = u_20 + alpha * (f_20 - A_20 @ u_20)
            cnt += 1
        print("{:4d}: {}".format(cnt, alpha))

# run sub questions
sub1()
sub2()
sub3()