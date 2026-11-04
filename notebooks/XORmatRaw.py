import numpy as np

xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ys = np.array([0, 1, 1, 0])

W1 = np.array([[0.5, -0.4],     # h1: w1, w2
               [-0.3, 0.8]])    # h2: w3, w4
b1 = np.array([0.1, -0.2])      # b1, b2
W2 = np.array([[0.7, -0.6]])    # output: w5, w6
b2 = np.array([0.05])           # b3

def sig(z):
    return 1 / (1 + np.exp(-z))

def forward(W1, b1, W2, b2, x):
    a1 = sig(W1 @ x + b1)
    a2 = sig(W2 @ a1 + b2)
    # z1 = W1 @ x + b1, print its shape
    # a1 = sig(z1)
    # z2 = W2 @ a1 + b2, print its shape
    # a2 = sig(z2)
    return a1, a2

def backward(a1, a2, y, x):
    err2 = a2 * (1 - a2) * 2 * (y - a2) # deritivite of sig(a2) * b = 1 * dertiive of (y - a2)^2    
    # a = answer so going one back is a1
    back = W2.T @ err2
    tW2 = np.outer(err2, a1)
    tb2 = err2
    err1 = back * a1 * (1 - a1)
    tW1 = np.outer(err1, x)
    tb1 = err1
    return tW2, tW1, tb1, tb2

lr = 0.5

for epoch in range(5001):
    total_loss = 0
    for x, y in zip(xs, ys):
        a1, a2 = forward(W1, b1, W2, b2, x)
        total_loss += (y - a2)**2
        tW2, tW1, tb1, tb2 = backward(a1, a2, y ,x)
        W2 += lr * tW2
        b2 += lr * tb2
        W1 += lr * tW1
        b1 += lr * tb1
        # forward
        # add this example's loss (y - a2)**2 to total_loss
        # backward
        # update all four: W2, b2, W1, b1
    if epoch % 1000 == 0:
        print(epoch, total_loss / 4)

for x in xs:
    print(x, forward(W1, b1, W2, b2, x)[1])