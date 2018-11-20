import time
import random
import math


N = 1000000

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a + b
    s += (time.time() - timer)
print(" +   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a - b
    s += (time.time() - timer)
print(" -   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a * b
    s += (time.time() - timer)
print(" *   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a / b
    s += (time.time() - timer)
print(" /   {:.10f}".format(s / N))

a = random.random()
c = 0
s = 0
sqrt = math.sqrt
for i in range(N):
    timer = time.time()
    c = sqrt(a)
    s += (time.time() - timer)
print("sqrt {:.10f}".format(s / N))

a = random.random() * 2 - 1
c = 0
s = 0
sin = math.sin
for i in range(N):
    timer = time.time()
    c = sin(a)
    s += (time.time() - timer)
print("sin  {:.10f}".format(s / N))

a = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = abs(a)
    s += (time.time() - timer)
print("abs  {:.10f}".format(s / N))