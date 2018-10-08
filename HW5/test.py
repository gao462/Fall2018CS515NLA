import numpy as np

A = np.random.random(size=(20, 20))

x = np.random.random(size=(len(A), 1))
x = x / np.linalg.norm(x)

for i in range(1000):
    x = A @ x
    x = x / np.linalg.norm(x)
    lam = x.T @ A @ x

    print(lam * x)
    print(A @ x)

    break

print(lam[0, 0])
print(np.linalg.eig(A)[0][0])
