import time
from matplotlib.image import imread
import numpy as np
import scipy.sparse as sparse


# load data
X1 = imread('image1.png').astype('float64')
X2 = imread('image2.png').astype('float64')
X3 = imread('image3.png').astype('float64')


# build a weight matrix
def crude_edge_detector(nin, nout, base):
    vecs = []
    for i in range(nin // 2):
        for j in range(nin // 2):
            mx = np.zeros((nin, nin))
            mx[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = base
            vecs.append(mx.reshape(-1))
    return np.array(vecs)

# build a sparse weight matrix
def sparse_crude_edge_detector(nin, nout, base):
    mx = sparse.lil_matrix((nout ** 2, nin ** 2))
    cnt = 0
    indices = np.arange(nin ** 2).reshape(nin, nin)
    for i in range(nin // 2):
        for j in range(nin // 2):
            i0, j0 = i * 2, j * 2
            for iv, jv in ((0, 0), (0, 1), (1, 0), (1, 1)):
                it = i0 + iv
                jt = j0 + jv
                vid = indices[it][jt]
                val = base[iv][jv]
                mx[cnt, vid] = val
            cnt += 1
    return mx

# build a dense weight matrix
def dense_crude_edge_detector(nin, nout, base):
    return sparse_crude_edge_detector(nin, nout, base).todense()


# buid a neural network
class Network(object):
    def __init__(self, base, type=None):
        if type is None:
            func = crude_edge_detector
        elif type == 'sparse':
            func = sparse_crude_edge_detector
        elif type == 'dense':
            func = dense_crude_edge_detector
        else:
            func = None
        self.W1 = crude_edge_detector(32, 16, base)
        self.W2 = crude_edge_detector(16, 8, base)
        self.W3 = np.ones((1, 8 * 8))

    def relu(self, x):
        x[x < 0] = 0
        return x

    def forward(self, x):
        x = self.relu(self.W1 @ x)
        x = self.relu(self.W2 @ x)
        x = self.relu(self.W3 @ x)
        return x

    def __call__(self, x):
        return self.forward(x)


# sub question 2
def sub2():
    print()
    print(np.trace(X1 + X2 + X3))

# sub question 4
def sub4():
    print()
    base = np.array([[1, -1], [-1, 1]])
    W1 = crude_edge_detector(32, 16, base)
    W2 = crude_edge_detector(16, 8, base)
    indices = np.where(W2[0] != 0)[0]
    vals = W2[0, indices]
    for idx, val in zip(indices, vals):
        print(1, idx + 1, val)

# sub question 5
def sub5():
    print()
    base = np.array([[1, -1], [-1, 1]])
    net = Network(base)
    print(net(X1.T.reshape(-1)))
    print(net(X2.T.reshape(-1)))
    print(net(X3.T.reshape(-1)))

# sub question 6
def sub6():
    print()
    base = np.array([[-1, 1], [1, -1]])
    net = Network(base)
    print(net(X1.T.reshape(-1)))
    print(net(X2.T.reshape(-1)))
    print(net(X3.T.reshape(-1)))

# sub question 7
def sub7():
    print()
    base = np.array([[1, -1], [-1, 1]])
    W1 = crude_edge_detector(32, 16, base)
    sW1 = sparse_crude_edge_detector(32, 16, base)
    dW1 = dense_crude_edge_detector(32, 16, base)
    print((sW1 == W1).all())
    print((dW1 == W1).all())
    print((sW1 == dW1).all())

# sub question 8
def sub8():
    print()
    X = X1.T.reshape(-1)
    base = np.array([[1, -1], [-1, 1]])
    sW1 = sparse_crude_edge_detector(32, 16, base)
    dW1 = dense_crude_edge_detector(32, 16, base)
    stime = 99
    dtime = 99
    for i in range(100):
        timer = time.time()
        _ = sW1 @ X
        stime = min(stime, time.time() - timer)
        timer = time.time()
        _ = dW1 @ X
        dtime = min(dtime, time.time() - timer)
    print("Sparse: {:.6f} sec".format(stime))
    print("Dense : {:.6f} sec".format(dtime))

# run sub questions
sub2()
sub4()
sub5()
sub6()
sub7()
sub8()