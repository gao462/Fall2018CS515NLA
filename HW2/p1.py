import numpy as np
import scipy


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


# sub question 1
def sub1():
    print()
    for focus in (137, 138, 139):
        cond1 = (candyland_matrix[:, 0] == focus)
        cond2 = (candyland_matrix[:, 0] != candyland_matrix[:, 1])
        cond3 = (candyland_matrix[:, 2] == 1)
        for dst, src, prob in candyland_matrix[cond1 & cond2 & cond3]:
            print("{:3d} ---> {:3d} : {:.6f}".format(int(src), int(dst), prob))

# sub question 2
def sub2():
    print()
    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    x = np.linalg.inv(i_mx - trans_mx.T) @ b
    print(x[139])
    longers = list(np.where(x > x[139])[0])
    if len(longers) > 0:
        for i in longers:
            print("[{:3d}] = {}".format(i + 1, x[i, 0]))
    else:
        print('No Longer Starting')
    return x[139] # expect length of starting from i = 140

# sub question 3
def sub3():
    print()
    T = trans_mx
    b = np.zeros((140, 1))
    b[139, 0] = 1
    k = 1
    p = T @ b
    S = k * p
    EPSILON = 1e-10
    while True:
        k += 1
        p = T @ p
        if k * np.linalg.norm(p) < EPSILON:
            break
        else:
            S = S + k * p
    print(S[133])
    return S[133] # expect length of reaching i = 134 (starting from 140)

# sub question 4
def sub4(ex2, ex3):
    print()
    EPSILON = 1e-6
    if np.fabs(ex2 - ex3) < EPSILON:
        print("Sub 2 == Sub 3 ({})".format(EPSILON))
    else:
        print("Sub 2 != Sub 3 ({})".format(EPSILON))

# sub question 5
def sub5():
    print()
    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    b[134, 0] = 0
    b[135, 0] = 0
    b[4, 0] = 0
    b[34, 0] = 0
    x = np.linalg.inv(i_mx - trans_mx.T) @ b
    print(x[139])
    longers = list(np.where(x > x[139])[0])
    if len(longers) > 0:
        for i in longers:
            print("[{:3d}] = {}".format(i + 1, x[i, 0]))
    else:
        print('No Longer Starting')
    return x[139] # expect length of starting from i = 140

# run sub questions
sub1()
ex2 = sub2()
ex3 = sub3()
sub4(ex2, ex3)
sub5()