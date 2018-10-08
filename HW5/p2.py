import numpy as np
np.random.seed(1234567890)


# +--------------------------------------------------
# | CandyLand From HW2
# +--------------------------------------------------


def candy_land():
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
    candyland_matrix = read_csv('candyland-matrix.csv')

    def matrix_from_liv(shape, liv):
        mx = np.zeros(shape)
        for i, j, val in liv:
            mx[int(i) - 1][int(j) - 1] = val
        return mx
    trans_mx = matrix_from_liv((140, 140), candyland_matrix)

    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    return i_mx - trans_mx.T, b


# +--------------------------------------------------
# | Iteration Methods
# +--------------------------------------------------


def loss(x, A, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)

def jacobi(A, b):
    D = np.diag(A)
    R = A - np.diag(D)
    D_inv = np.diag(1 / D)
    x = b.copy()
    cnt = 0
    while loss(x, A, b) > 1e-4:
        x = D_inv @ (b - R @ x)
        cnt += 1
    return x, cnt

def gauss_seidel(A, b):
    x = b.copy()
    cnt = 0
    while loss(x, A, b) > 1e-4:
        for i in range(len(x)):
            sigma = b[i, 0]
            for j in range(0, len(x)):
                if j == i:
                    continue
                else:
                    sigma -= A[i, j] * x[j, 0]
            x[i, 0] = sigma / A[i, i]
        cnt += 1
    return x, cnt

# run problem
print()
A, b = candy_land()
x, cnt = jacobi(A, b)
print(cnt)
x, cnt = gauss_seidel(A, b)
print(cnt)