import numpy as np


# sub question 1
def sub1():
    print()
    a = np.array([
        [1, 1, 2],
        [3, 5, 8],
        [13, 21, 34],
    ])
    b = np.array([
        [1, -2, 3],
        [-4, 5, -6],
        [7, -8, 9],
    ])
    print(a @ b)

# sub question 2
def sub2():
    print()
    x = np.ones((1000, 1))
    y = np.arange(1, 1000 + 1)[: , np.newaxis]
    print(x.T @ y)

# sub question 3
def sub3():
    print()
    x = np.array([1.5, 2, -3])[: , np.newaxis]
    e = np.ones((4, 1))
    print(x.shape, e.shape)
    print(e @ x.T)
    print(x @ e.T)

# sub question 4
def sub4():
    print()
    x = np.array([-5, 4, 2])[: , np.newaxis]
    e_1 = np.zeros((3, 1))
    e_1[0] = 1
    e_3 = np.zeros((3, 1))
    e_3[2] = 1
    print(e_1 @ x.T)
    print(x @ e_3.T)

# run sub questions
sub1()
sub2()
sub3()
sub4()