"""
Module 1 — Autograd Engine (starter)

Fill the TODOs. When all tests in test.py pass, you've built backprop.

Rule: do NOT look at solution.py until tests pass.
"""

import math


class Value:
    """A scalar value that builds a computation graph and can backprop."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        # internal: bookkeeping for the autograd graph
        self._prev = set(_children)
        self._op = _op
        # called during backward() to push grad to parents
        self._backward = lambda: None

    # ---------- forward ops ----------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO: d(self+other)/d(self) = 1, d(...)/d(other) = 1
            # Remember: ACCUMULATE with +=, do not overwrite.
            pass

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: d(self*other)/d(self) = other.data
            #       d(self*other)/d(other) = self.data
            pass

        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only supports int/float exponents"
        out = Value(self.data ** n, (self,), f"**{n}")

        def _backward():
            # TODO: d(self**n)/d(self) = n * self.data ** (n - 1)
            pass

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # TODO: derivative is 1 if self.data > 0 else 0
            pass

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO: d(tanh(x))/dx = 1 - tanh(x)^2  (use out.data, it IS tanh(x))
            pass

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # TODO: d(exp(x))/dx = exp(x) = out.data
            pass

        out._backward = _backward
        return out

    # ---------- the big one ----------

    def backward(self):
        # TODO:
        # 1) Build a topological ordering of the graph reachable from `self`.
        # 2) Set self.grad = 1.0 (dL/dL = 1).
        # 3) Walk topo in REVERSE order, calling node._backward() on each.
        topo = []
        visited = set()

        def build_topo(v):
            # TODO: DFS, append v AFTER visiting children
            pass

        build_topo(self)
        # TODO: set seed gradient and walk in reverse
        # for node in reversed(topo): node._backward()
        pass

    # ---------- syntactic sugar (mostly free once you have add/mul) ----------

    def __neg__(self):       # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):   # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other  ==  self * other**-1
        return self * (other ** -1 if isinstance(other, (int, float)) else other ** -1)

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == "__main__":
    # Sanity smoke test (not the real test). The real test is in test.py.
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b
    e = d + c
    f = e * Value(2.0)
    f.backward()
    print(f, a.grad, b.grad, c.grad)  # expect a.grad=-6, b.grad=4, c.grad=2
