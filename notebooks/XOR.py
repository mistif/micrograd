import math, random

xs = [(0, 0), (0, 1), (1, 0), (1, 1)]
ys = [0, 1, 1, 0]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def forward(x, w1, w2, w3, w4, w5, w6, b1, b2, b3):
    x1, x2 = x
    z1 = x1 * w1 + x2 * w2 + b1
    z2 = x1 * w3 + x2 * w4 + b2
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    z3 = a1 * w5 + a2 * w6 + b3
    a3 = sigmoid(z3)
    # z1, a1 (hidden 1)
    # z2, a2 (hidden 2)
    # z3, a3 (output)
    return z1, z2, z3, a1, a2, a3

def backward(x, y, a1, a2, a3, w5, w6):
    x1, x2 = x
    s1 = a1 * (1 - a1)
    s2 = a2 * (1 - a2)
    s3 = a3 * (1 - a3)

    err3 = s3 * 2 * (a3 - y) # dc/dz3 
    gw5 = a1 * err3
    gw6 = a2 * err3
    gb3 = err3

    dC_da1 = w5 * err3
    dC_da2 = w6 * err3

    gw1 = dC_da1 * s1 * x1
    gw2 = dC_da1 * s1 * x2
    gw3 = dC_da2 * s2 * x1
    gw4 = dC_da2 * s2 * x2
    gb1 = s1 * dC_da1
    gb2 = s2 * dC_da2
    # slopes s1, s2, s3
    # output neuron: err3, gw5, gw6, gb3
    # blame back: dC_da1, dC_da2
    # hidden 1: gw1, gw2, gb1
    # hidden 2: gw3, gw4, gb2
    return gw1, gw2, gb1, gw3, gw4, gb2, gw5, gw6, gb3

w1, w2, w3, w4, w5, w6 = [random.uniform(-1, 1) for _ in range(6)]
b1, b2, b3 = 0.0, 0.0, 0.0
lr = 1.0

for step in range(10000):
    totals = [0] * 9
    for x, y in zip(xs, ys):
        # forward, backward, add grads to totals
        z1, z2, z3, a1, a2, a3 = forward(x, w1, w2, w3, w4, w5, w6, b1, b2, b3)
        grads = backward(x, y, a1, a2, a3, w5, w6)
        for i in range(9):
            totals[i] += grads[i]
    

    g = [t / len(xs) for t in totals]
    w1 -= g[0] * lr
    w2 -= g[1] * lr
    b1 -= g[2] * lr
    w3 -= g[3] * lr
    w4 -= g[4] * lr
    w5 -= g[6] * lr
    b2 -= g[5] * lr
    w6 -= g[7] * lr
    b3 -= g[8] * lr
    # update all 9 params

for x in xs:
    print(x, round(forward(x, w1, w2, w3, w4, w5, w6, b1, b2, b3)[5], 3))