import math


def f(a, b, c, x):
    if x is None:
        return None
    else:
        return a * (x ** 2) + b * x + c

def sign(v):
    if v > 0:
        return 1
    elif v < 0:
        return -1
    else:
        return 0

def bisect(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    l0 = (-b - max(2 * t, 1)) / (2 * a)
    m0 = -b / (2 * a)
    r0 = (-b + max(2 * t, 1)) / (2 * a)
    if sign(f(a, b, c, l0)) != sign(f(a, b, c, m0)):
        x1 = l0
        x2 = m0
        m = (x1 + x2) / 2
        while (x1 < m) and (m < x2):
            if sign(f(a, b, c, m)) != sign(f(a, b, c, x2)):
                x1 = m
            else:
                x2 = m
            m = (x1 + x2) / 2
        r1 = m
    else:
        r1 = None
    if sign(f(a, b, c, m0)) != sign(f(a, b, c, r0)):
        x1 = m0
        x2 = r0
        m = (x1 + x2) / 2
        while (x1 < m) and (m < x2):
            if sign(f(a, b, c, x1)) != sign(f(a, b, c, m)):
                x2 = m
            else:
                x1 = m
            m = (x1 + x2) / 2
        r2 = m
    else:
        r2 = None
    return r1, r2

def quadratic(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    r1 = (-b - t) / (2 * a)
    r2 = (-b + t) / (2 * a)
    return r1, r2

def citardauq(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    if -b + t == 0:
        r1 = None
    else:
        r1 = (2 * c) / (-b + t)
    if -b - t == 0:
        r2 = None
    else: 
        r2 = (2 * c) / (-b - t)
    return r1, r2

def fix(a, b, c, r1, r2):
    e1, e2 = f(a, b, c, r1), f(a, b, c, r2)
    if e1 is None and e2 is None:
        pass
    elif e1 is None or math.fabs(e1) > math.fabs(e2):
        e3 = f(a, b, c, c / a / r2)
        if e1 is None or math.fabs(e3) < math.fabs(e1):
            r1 = c / a / r2
        else:
            pass
    elif e2 is None or math.fabs(e2) > math.fabs(e1):
        e3 = f(a, b, c, c / a / r1)
        if e2 is None or math.fabs(e3) < math.fabs(e2):
            r2 = c / a / r1
        else:
            pass
    else:
        pass
    return r1, r2


# sub question 1
def q1(do_fix=True):
    print()
    print('-----' * 22)
    for a, b, c in (
        (1, 3, 2), (1, 3, 9 / 4 - 1e-20),
        (1e-20, 3, 2), (1e-20, 3, 2e20),
        (1e-20, 3, 2e-20), (1e-20, 3e-20, 2e-20)):
        r11, r12 = bisect(a, b, c)
        if do_fix:
            r11, r12 = fix(a, b, c, r11, r12)
        else:
            pass
        e11, e12 = f(a, b, c, r11), f(a, b, c, r12)
        r21, r22 = quadratic(a, b, c)
        if do_fix:
            r21, r22 = fix(a, b, c, r21, r22)
        else:
            pass
        e21, e22 = f(a, b, c, r21), f(a, b, c, r22)
        r31, r32 = citardauq(a, b, c)
        if do_fix:
            r31, r32 = fix(a, b, c, r31, r32)
        else:
            pass
        e31, e32 = f(a, b, c, r31), f(a, b, c, r32)
        s1 = "Bisection: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e11 is None else ('%.16f' % math.fabs(e11))[0:20],
                'NaN' if r11 is None else ('%.16f' % math.fabs(r11))[0:20],
                'NaN' if e12 is None else ('%.16f' % math.fabs(e12))[0:20],
                'NaN' if r12 is None else ('%.16f' % math.fabs(r12))[0:20])
        s2 = "Quadratic: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e21 is None else ('%.16f' % math.fabs(e21))[0:20],
                'NaN' if r21 is None else ('%.16f' % math.fabs(r21))[0:20],
                'NaN' if e22 is None else ('%.16f' % math.fabs(e22))[0:20],
                'NaN' if r22 is None else ('%.16f' % math.fabs(r22))[0:20])
        s3 = "Citardauq: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e31 is None else ('%.16f' % math.fabs(e31))[0:20],
                'NaN' if r31 is None else ('%.16f' % math.fabs(r31))[0:20],
                'NaN' if e32 is None else ('%.16f' % math.fabs(e32))[0:20],
                'NaN' if r32 is None else ('%.16f' % math.fabs(r32))[0:20])
        res = [
            (max(math.fabs(e11), math.fabs(e12)), math.fabs(e11) + math.fabs(e12), s1),
            (max(math.fabs(e21), math.fabs(e22)), math.fabs(e21) + math.fabs(e22), s2),
            (max(math.fabs(e31), math.fabs(e32)), math.fabs(e31) + math.fabs(e32), s3)]
        for i, (_, _, s) in enumerate(sorted(res, key=lambda x: (x[0], x[1]))):
            print("{} {}".format(i, s))
        print('-----' * 22)

# run sub questions
q1(True)