def raw_sum(x):
    return sum(x)

def perm_sum(x):
    return sum(sorted(x))

def kahan_sum(x):
    s, c = 0, 0
    y, t = None, None
    for itr in x:
        y = itr - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# sub question 4
def p4():
    x = [1e16] + [1] * int(1e4)
    print("Raw    : {:f}".format(raw_sum(x)))
    print("Permute: {:f}".format(perm_sum(x)))
    print("Kahan  : {:f}".format(kahan_sum(x)))

# run sub questions
p4()