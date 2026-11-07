# ML from scratch

My path from zero to building neural networks by hand, ahead of building my own LLM.

## What's here

| Notebook | What I built | Result |
|---|---|---|
| `micrograd.ipynb` | Autograd engine (Value class, backprop, Neuron/Layer/MLP), following Karpathy | Loss 4.72 → 0.06 in 20 steps |
| `XOR.py`, `XORmatrices.py` | XOR network in NumPy, backprop by hand | Solves XOR |
| `mnist_numpy.ipynb` | 784-16-16-10 digit recognizer in pure NumPy: my own forward, backward, softmax and one-hot | Sigmoid 94.3%, ReLU 93.9% |
| `mnist_digits.ipynb` | Same net in PyTorch, then bigger experiments | 97.5% (512-512, ReLU) |
| `pytorch_basics.ipynb` | PyTorch Learn the Basics tutorial | |
| `01_first_model`, `02_house_prices` | First models with pandas and scikit-learn | |

## Things I learned the hard way
- ReLU beats sigmoid when the net is big, because its slope is 1 instead of saturating.
- Too high a learning rate can kill every ReLU neuron for good (stuck at 10%).
- Softmax needs the row max subtracted first, or exp() overflows.

## Run it
