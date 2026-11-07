import math
import random

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [0, 1, 1, 0]

b1, b2, b3, w1, w2, w3, w4, w5, w6 = 0.1, -0.2, 0.05, 0.5, -0.4, -0.3, 0.8, 0.7, -0.6
lr = .5
def sig(z):
    return 1 / (1 + math.exp(-z))

def s(a):
     return a * (1 - a)

def forward(b1, b2, b3, w1, w2, w3, w4, w5, w6, x):
        x1, x2 = x
        z1 = w1 * x1 + w2 * x2 + b1
        z2 = w3 * x1 + w4 * x2 + b2
        a1 = sig(z1)
        a2 = sig(z2)
        z3 = a1 * w5 + a2 * w6 + b3
        a3 = sig(z3)

        return a1, a2, a3, z1, z2, z3

def backward(a1, a2, a3, w5, w6, y, x):
    #return all b's and w's
    x1,x2 = x
    s1 = s(a1)
    s2 = s(a2)
    s3 = s(a3)

    err3 = s3 * 2 * (a3 - y)
    gw5 = err3 * a1
    gw6 = err3 * a2

    err2 = s2 * w6 * err3
    gw4 = err2 * x2
    gw3 = err2 * x1

    err1 = s1 * w5 * err3
    gw2 = err1 * x2
    gw1 = err1 * x1

    gb1, gb2, gb3 = err1, err2, err3

    return gb1, gb2, gb3, gw1, gw2, gw3, gw4, gw5, gw6

lr = 0.5

for epoch in range(10001):
    total = 0
    for x, y in zip(xs, ys):
        a1, a2, a3, z1, z2, z3 = forward(b1, b2, b3, w1, w2, w3, w4, w5, w6, x)
        total += (a3 - y) ** 2

        gb1, gb2, gb3, gw1, gw2, gw3, gw4, gw5, gw6 = backward(a1, a2, a3, w5, w6, y, x)

        b1 -= lr * gb1
        b2 -= lr * gb2
        b3 -= lr * gb3
        w1 -= lr * gw1
        w2 -= lr * gw2
        w3 -= lr * gw3
        w4 -= lr * gw4
        w5 -= lr * gw5
        w6 -= lr * gw6

    if epoch % 2000 == 0:
        print(epoch, round(total, 4))

for x in xs:
    print(x, round(forward(b1, b2, b3, w1, w2, w3, w4, w5, w6, x)[2], 3))

print(forward(b1, b2, b3, w1, w2, w3, w4, w5, w6, xs[0]))