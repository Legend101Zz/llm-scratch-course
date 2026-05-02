# Module 6 — PyTorch Crash Course (30 min)

> ⏱️ 30 minutes. The minimum PyTorch you need to never feel lost again.

## Mental Model: PyTorch IS Your Autograd Engine, on Steroids

Compare to Module 1:

| Module 1 (yours)         | PyTorch                          |
|--------------------------|----------------------------------|
| `Value(2.0)`             | `torch.tensor(2.0, requires_grad=True)` |
| scalar                   | n-dim tensor (CPU/GPU)           |
| `out._backward = ...`    | C++ autograd engine, automatic   |
| `value.backward()`       | `loss.backward()`                |
| `p.data -= lr * p.grad`  | `optimizer.step()` + `zero_grad()` |
| `Neuron`, `Layer`, `MLP` | `nn.Linear`, `nn.Module`         |

Same ideas, optimized + GPU. If you grokked Module 1, the only new things are:
1. **Tensor shapes & broadcasting**.
2. **`nn.Module` API** (`__init__` for params, `forward` for computation).
3. **Optimizer + DataLoader plumbing**.

## The 7 things you actually need

### 1. Tensors

```python
import torch
x = torch.tensor([[1., 2.], [3., 4.]])   # shape (2, 2)
y = torch.zeros(3, 4)
z = torch.randn(2, 3, 4)                 # standard normal
x.shape, x.dtype, x.device
x.to("cuda")                             # move to GPU
```

### 2. Broadcasting

```python
x = torch.randn(4, 1)
y = torch.randn(1, 3)
(x + y).shape   # (4, 3)
```

Two dims are compatible if equal, or one is 1, or one is missing.

### 3. Autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()       # populates x.grad
print(x.grad)      # 6.0
```

### 4. `nn.Module`

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 4)   # weight (4, 2), bias (4,)
        self.l2 = nn.Linear(4, 1)
    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))
```

`nn.Module` collects all submodules' parameters via `model.parameters()`.

### 5. Loss + Optimizer

```python
import torch.nn.functional as F

loss = F.mse_loss(pred, target)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
# training step:
opt.zero_grad()
loss.backward()
opt.step()
```

### 6. Cross-entropy for LMs

```python
# logits: (B, T, V)   targets: (B, T)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
```
PyTorch's `cross_entropy` does logits → softmax → NLL in one numerically-stable shot. Don't compute softmax yourself before this.

### 7. Useful debugging

```python
print(x.shape)            # always
torch.manual_seed(0)      # reproducibility
with torch.no_grad():     # for inference, no graph built
    ...
```

## Exercise (15 min)

Open [`xor_torch.py`](xor_torch.py). Rewrite Module 2's XOR MLP in PyTorch. Goal: feel the difference. Half the lines, GPU-ready, faster.

✅ Next: [Module 7 — Mini-GPT in PyTorch](../07_gpt_pytorch/README.md). The big one.
