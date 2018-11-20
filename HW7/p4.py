import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)

def loss(x, A, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    
def gauss_seidel(A, b):
    x = b.copy()
    cnt = 0
    old_loss = loss(x, A, b)
    rate = None
    while old_loss > 1e-4 and cnt < 100:
        for i in range(len(x)):
            sigma = b[i, 0]
            for j in range(0, len(x)):
                if j == i:
                    continue
                else:
                    sigma -= A[i, j] * x[j, 0]
            x[i, 0] = sigma / A[i, i]
        cnt += 1
        new_loss = loss(x, A, b)
        rate = new_loss / old_loss
        old_loss = new_loss
    return loss(x, A, b)

results = []
for N in (2, 3, 4):
    b = np.random.normal(size=(N, 1))
    for i in range(1000):
        A = np.ones(shape=(N, N))
        for r in range(A.shape[0]):
            for c in range(r + 1, A.shape[1]):
                if r == c:
                    continue
                else:
                    pass
                val = (np.random.random() - 0.5) * 2 * np.exp(np.random.randint(-20, 2))
                A[r, c] = val
                A[c, r] = val
        cond, metric = np.linalg.cond(A), gauss_seidel(A, b)
        if np.all(np.linalg.eig(A)[0] > 0) and np.all(np.diag(A) == 1) and np.all(A.T == A):
            results.append((cond, metric))
        else:
            pass

results = sorted(results, key=lambda x: x[0])
results = np.array(results, dtype=float)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(np.log(results[:, 0]), np.log(results[:, 1]), marker='.')
ax.set_xlabel('Log Condition Number')
ax.set_ylabel('Log Relative Residual (100 Iteration)')
fig.savefig('p4.png')