"""Module 2 — MLP on top of your autograd. Fill TODOs."""

import random
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_autograd"))
from solution import Value  # use the reference engine so you don't debug both at once

random.seed(1337)


class Neuron:
    def __init__(self, n_in, nonlin=True):
        # TODO: create n_in weights as Values randomly in [-1, 1], plus a bias Value.
        self.w = []         # list[Value]
        self.b = None       # Value
        self.nonlin = nonlin

    def __call__(self, x):
        # x: list[Value or number]
        # TODO: compute sum(w_i * x_i) + b, then tanh() if self.nonlin
        pass

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        # TODO: list of n_out Neurons, each taking n_in inputs.
        self.neurons = []

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, hidden_sizes):
        # hidden_sizes example: [4, 4, 1]  → two hidden + 1-output layer
        sizes = [n_in] + hidden_sizes
        self.layers = []
        # TODO: build len(hidden_sizes) Layers. Last layer is linear (nonlin=False).

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


# ----- training loop on XOR -----

if __name__ == "__main__":
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [0, 1, 1, 0]

    model = MLP(2, [4, 4, 1])
    print("params:", len(model.parameters()))

    lr = 0.1
    for step in range(500):
        # forward + loss
        # TODO: compute predictions, then MSE loss as a single Value
        # preds = [model(x) for x in xs]
        # loss = sum((p - y)**2 for p, y in zip(preds, ys)) * (1.0 / len(xs))

        # zero grads
        # TODO: for p in model.parameters(): p.grad = 0.0

        # backward
        # TODO: loss.backward()

        # update
        # TODO: for p in model.parameters(): p.data -= lr * p.grad

        if step % 50 == 0:
            # print(f"step {step:4d}  loss {loss.data:.6f}")
            pass

    # print final predictions
    # for x, y in zip(xs, ys): print(x, "→", model(x).data, "(target", y, ")")
