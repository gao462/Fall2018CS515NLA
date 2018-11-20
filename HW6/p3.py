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

# sub question 1
def sub1():
    # show image
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_data[67 - 1].reshape(64, 64).T, cmap='gray')
    fig.savefig('image67.png')
    plt.close(fig)

# sub question 2
def sub2():
    # construct A
    A_list = []
    for i in range(len(img_data)):
        if (i + 1) % 2 == 0:
            A_list.append(img_data[i])
        else:
            pass
    A = np.array(A_list).T

    # construct and solve b for each test case
    B_list = []
    ind_list = []
    for i in range(len(img_data)):
        if (i + 1) % 2 == 1:
            B_list.append(img_data[i])
            ind_list.append(i)
        else:
            pass
    B = np.array(B_list).T
    C, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)
    print("Worst: {:3d}".format(ind_list[residuals.argmin()] + 1))
    print("Best : {:3d}".format(ind_list[residuals.argmax()] + 1))


# sub question 3
def sub3():
    # construct A
    A_list = []
    for i in range(len(img_data)):
        A_list.append(img_data[i])
    A = np.array(A_list).T

    # construct b
    b = img_data[67 - 1].reshape(-1, 1)

    # test sigma
    def test(sigma, range1, range2):
        cnt1 = 0
        cnt2 = 0
        for i in range(100):
            nb = b + sigma * np.random.normal(size=b.shape)
            c, residual, _, _ = np.linalg.lstsq(A, nb, rcond=None)
            res = np.fabs(c).argmax()
            print(np.fabs(c).max())
            if res in range1:
                pass
            else:
                cnt1 += 1
            if res in range2:
                pass
            else:
                cnt2 += 1
        return cnt1 < 25, cnt2 < 25

    # add noise
    flag1, flag2 = test(191, [66], list(range(66, 77)))
    print(flag1, flag2)
    flag1, flag2 = test(192, [66], list(range(66, 77)))
    print(flag1, flag2)

# run sub questions
# sub1()
# sub2()
sub3()