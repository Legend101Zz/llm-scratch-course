"""Module 2 — MLP solution."""

import random, importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "autograd_solution",
    os.path.join(os.path.dirname(__file__), "..", "01_autograd", "solution.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Value = _mod.Value

random.seed(1337)


class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, hidden_sizes):
        sizes = [n_in] + hidden_sizes
        self.layers = [
            Layer(sizes[i], sizes[i+1], nonlin=(i != len(hidden_sizes) - 1))
            for i in range(len(hidden_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


if __name__ == "__main__":
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [0, 1, 1, 0]
    model = MLP(2, [4, 4, 1])

    for step in range(500):
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) * (1.0 / len(xs))
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        lr = 0.1
        for p in model.parameters():
            p.data -= lr * p.grad
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.data:.6f}")

    for x, y in zip(xs, ys):
        print(x, "→", round(model(x).data, 3), "(target", y, ")")
