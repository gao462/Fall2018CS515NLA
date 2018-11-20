import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)


# load raw file
img_file = open('Yale_64.csv', 'r')
lbl_file = open('Yale_64_ids.csv', 'r')
img_content = list(img_file.readlines())
lbl_content = list(lbl_file.readlines())
img_file.close()
lbl_file.close()
img_data = []
lbl_data = []
for img_line, lbl_line in zip(img_content, lbl_content):
    image = img_line.strip().split(',')
    image = np.array([int(itr) for itr in image])
    label = int(lbl_line.strip())
    img_data.append(image)
    lbl_data.append(label)
img_mx = np.array(img_data, dtype=float).T
lbl_mx = np.array(lbl_data, dtype=int).reshape(1, -1)


# sub question 2
def sub2():
    print()
    n = img_mx.shape[0]
    avg_mx = np.ones(shape=(n, n)) / n
    img_cmx = img_mx - avg_mx @ img_mx

# sub question 3
def sub3():
    print()
    img_cmx = img_mx - img_mx.mean(axis=0)
    u_mx, s, vt_mx = np.linalg.svd(img_cmx)
    print(s.max())

# sub question 4
def sub4():
    print()
    img_cmx = img_mx - img_mx.mean(axis=0)
    u_mx, s, vt_mx = np.linalg.svd(img_cmx)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(u_mx[0, :].reshape(64, 64).T, cmap='gray')
    fig.savefig('u1.png')
    plt.close(fig)

# run sub questions
sub2()
sub3()
sub4()