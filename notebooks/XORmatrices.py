import numpy as np
np.random.seed(0)

X = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
Y = [0, 1, 1, 0]

W1 = np.random.randn(16, 2)
b1 = np.zeros((16, 1))
W2 = np.random.randn(1, 16)
b2 = np.zeros((1, 1))

def sig(z):
    return 1 / ( 1 + np.exp(-z))

lr = 0.5
for epoch in range(5000):
    tW1 = np.zeros_like(W1); tb1 = np.zeros_like(b1) # tW1 = all gws and W1 is all the weights
    tW2 = np.zeros_like(W2); tb2 = np.zeros_like(b2)

    for x, y in zip(X, Y):
        # forward
        a1 = sig(W1 @ x + b1) # output from 16 neruons
        a2 = sig(W2 @ a1 + b2) # answer

        # backward
        err2 = a2 * (1 - a2) * 2 * (a2 - y)            # same as your err3 a1 * (1 - a1) * 2(a - y)
        tW2 += err2 @ a1.T            # hint: err2 @ (something).T
        tb2 += err2
        err1 = W2.T @ err2 * a1  * (1 - a1)        # hint: W2.T @ err2, then times sig'
        tW1 += err1 @ x.T
        tb1 += err1

    W1 -= lr * tW1 / len(Y)
    b1 -= lr * tb1 / len(Y)
    W2 -= lr * tW2 / len(Y)
    b2 -= lr * tb2 / len(Y)

for x in X:
    a1 = sig(W1 @ x + b1)
    a2 = sig(W2 @ a1 + b2)
    print(x.T, round(a2.item(), 3))          # run forward again and print a2