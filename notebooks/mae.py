import math

xs = [(0, 0), (0, 1), (1, 0), (1, 1)]
ys = [0, 1, 1, 1]

w1, b1 = 0.5, 0.0   # hidden neuron
w2, b2 = 0.5, 0.0   # output neuron
lr = 1.0

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def forward(x, w1, b1, w2, b2):
    z1 = (w1 * x) + b1
    a1 = sigmoid(z1)
    z2 = (w2 * a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2
    # run x through the hidden neuron, then the output neuron
    # return z1, a1, z2, a2

def backward(x, y, w2, z1, a1, z2, a2):
    # return gw1, gb1, gw2, gb2 for this one point
    # output layer first, then pass the blame back to the hidden layer
    s1 = sigmoid(z1) * (1-sigmoid(z1))
    s2 = sigmoid(z2) * (1-sigmoid(z2))
    dC_da1 = w2 * s2 * 2 * (a2 - y)
    gw2 = a1 * s2 * 2 * (a2 - y)
    gb2 = s2 * 2 *(a2 - y)
    gw1 = x * s1 * dC_da1
    gb1 = s1 * dC_da1

    return gw2, gb2, gw1, gb1

    pass

for step in range(5000):
    # for each point: forward, then backward, and add up the 4 gradients
    # then update all 4 parameters with the averages
    total_gw2, total_gb2, total_gw1, total_gb1 = 0, 0 ,0 ,0
    for x,y in zip(xs, ys):

        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
        gw2, gb2, gw1, gb1 = backward(x, y, w2, z1, a1, z2,  a2)
        total_gb1 += gb1
        total_gb2 += gb2
        total_gw1 += gw1
        total_gw2 += gw2
  

    w1 -= lr * total_gw1 / len(xs)
    w2 -= lr * total_gw2 / len(xs)
    b1 -= lr * total_gb1 / len(xs)
    b2 -= lr * total_gb2 / len(xs)


    

for x in xs:
    z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
    print(x, round(a2, 3))