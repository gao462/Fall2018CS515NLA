import numpy as np
import scipy.sparse as sparse
np.random.seed(1234567890)


# get csc format from dense matrix
def dense_to_csc(mx):
    colptr = []
    rowptr = []
    nzval = []
    for cid in range(mx.shape[1]):
        colptr.append(len(rowptr))
        for rid in range(mx.shape[0]):
            if mx[rid, cid] != 0:
                rowptr.append(rid)
                nzval.append(mx[rid, cid])
            else:
                pass
    colptr.append(len(rowptr))
    return colptr, rowptr, nzval
mx = sparse.random(5, 6, density=0.4).todense()
colptr, rowval, nzval = dense_to_csc(mx)

# Sparse matrix-transpose multiplication by a vector
def csc_transpose_matvec(colptr, rowval, nzval, m, n, x):
    y = np.zeros((len(colptr) - 1, 1))
    for cid, (begin, end) in enumerate(zip(colptr[:-1], colptr[1:])):
        for rid, val in zip(rowval[begin:end], nzval[begin:end]):
            y[cid] += val * x[rid, 0]
    return y

# Row-inner-product
def csc_row_projection(colptr, rowval, nzval, m, n, i, x):
    y = np.zeros((1, 1))
    for cid, (begin, end) in enumerate(zip(colptr[:-1], colptr[1:])):
        for rid, val in zip(rowval[begin:end], nzval[begin:end]):
            if rid == i:
                y[0, 0] += val * x[cid, 0]
            else:
                pass
    return y

# Column-inner-product
def csc_column_projection(colptr, rowval, nzval, m, n, i, x):
    y = np.zeros((1, 1))
    begin, end = colptr[i], colptr[i + 1]
    for rid, val in zip(rowval[begin:end], nzval[begin:end]):
        y[0, 0] += val * x[rid, 0]
    return y

# sub question 1 test
def sub1():
    print()
    vec = np.random.normal(size=(5, 1))
    y1 = mx.T @ vec
    y2 = csc_transpose_matvec(colptr, rowval, nzval, None, None, vec)
    EPSILON = 1e-8
    print(np.fabs(y1 - y2).max() < EPSILON)

# sub question 2 test
def sub2():
    print()
    res = []
    for i in range(mx.shape[0]):
        vec = np.random.normal(size=(mx.shape[1], 1))
        y1 = mx[i, :] @ vec
        y2 = csc_row_projection(colptr, rowval, nzval, None, None, i, vec)
        EPSILON = 1e-8
        res.append(np.fabs(y1 - y2).max() < EPSILON)
    print(not False in res)

# sub question 3 test
def sub3():
    print()
    res = []
    for i in range(mx.shape[1]):
        vec = np.random.normal(size=(mx.shape[0], 1))
        y1 = mx[:, i].T @ vec
        y2 = csc_column_projection(colptr, rowval, nzval, None, None, i, vec)
        EPSILON = 1e-8
        res.append(np.fabs(y1 - y2).max() < EPSILON)
    print(not False in res)

# run sub question tests
sub1()
sub2()
sub3()