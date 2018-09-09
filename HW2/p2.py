import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# sub question 1
def sub1():
    print()
    N = 3
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            print("[{:2d}]: ({:1d}, {:1d})".format(idx_mx[i, j], i, j), end='')
            if i == 0 or i == N or j == 0 or j == N:
                print(' = 0')
            else:
                mx_op_str = []
                vec_op_str = []
                for (dx_i, dx_j, val) in ops:
                    mx_op_str.append("{:2d} * ({:1d}, {:1d})".format(
                                        val, i + dx_i, j + dx_j))
                    vec_op_str.append("{:2d} * [{:2d}]".format(
                                        val, idx_mx[i + dx_i, j + dx_j]))
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
                mx_op_str = ' + '.join(mx_op_str)
                vec_op_str = ' + '.join(vec_op_str)
                print(" = {} = {}".format(mx_op_str, vec_op_str))
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    print(A)

# sub question 2
def laplacian(N, f):
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    fvec = np.zeros(((N + 1) ** 2, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            fvec[idx_mx[i, j] - 1, 0] = f(i / N, j / N)
            if i == 0 or i == N or j == 0 or j == N:
                pass
            else:
                for (dx_i, dx_j, val) in ops:
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    return A, fvec

# sub question 3
def sub3():
    print()
    A, fvec = laplacian(10, lambda i, j: 1)
    uvec = np.linalg.lstsq(A, fvec, rcond=None)[0]
    N = int(np.sqrt(len(uvec))) - 1
    rows = np.tile(np.arange(0, N + 1), (N + 1, 1)).T
    cols = np.tile(np.arange(0, N + 1), (N + 1, 1))
    vals = uvec.reshape(N + 1, N + 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(rows, cols, vals, cmap=cm.YlGnBu)
    fig.savefig('p2.png')
    plt.show(fig)

# run sub questions
sub1()
sub3()