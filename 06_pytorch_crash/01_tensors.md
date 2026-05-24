# 6.1 — Tensors, Top to Bottom

> Goal: by the end of this doc, you can confidently create, index, reshape, and broadcast tensors of any shape — and know why each op exists.

A **tensor** is a typed, n-dimensional array that lives on a specific device (CPU or a particular GPU). Almost everything in PyTorch is a tensor — model weights, gradients, inputs, outputs, intermediate activations.

---

## 1. Creating Tensors

```python
import torch

# from Python data
t = torch.tensor([[1., 2.], [3., 4.]])     # (2, 2), float32

# fixed shape
torch.zeros(3, 4)                          # all zeros
torch.ones(3, 4)                           # all ones
torch.full((3, 4), 7.0)                    # all 7s
torch.empty(3, 4)                          # uninitialized — fastest, contents are garbage

# random
torch.rand(2, 3)                           # uniform [0, 1)
torch.randn(2, 3)                          # standard normal (mean 0, std 1)
torch.randint(0, 10, (2, 3))               # uniform integers in [0, 10)

# ranges / structure
torch.arange(10)                           # 0, 1, ..., 9
torch.arange(0, 1, 0.1)                    # start, stop, step
torch.linspace(0, 1, 5)                    # 5 evenly spaced from 0 to 1
torch.eye(3)                               # 3×3 identity

# from another tensor (preserves dtype/device by default)
torch.zeros_like(t)
torch.ones_like(t)
torch.randn_like(t)
```

### Inspecting tensors

```python
t.shape         # torch.Size([2, 2])
t.size()        # equivalent
t.size(0)       # int — size of dim 0
t.dtype         # torch.float32
t.device        # device(type='cpu')
t.numel()       # number of elements = 4
t.ndim          # number of dimensions = 2
t.element_size() # bytes per element
t.requires_grad # is autograd tracking?
```

---

## 2. Dtypes — Why They Matter

| dtype | Bytes | What for |
|---|---|---|
| `torch.float32` (`float`) | 4 | Default for math. Train weights in this. |
| `torch.float64` (`double`) | 8 | Scientific computing. Slow on GPUs. Avoid for DL. |
| `torch.float16` (`half`) | 2 | Mixed-precision training (with AMP). Faster, less memory. |
| `torch.bfloat16` | 2 | Google's variant — same range as float32, less precision. Preferred over float16 on Ampere+ GPUs. |
| `torch.int64` (`long`) | 8 | Default integer. **Indices and token IDs use this.** |
| `torch.int32` (`int`) | 4 | Less common; sometimes for mem savings. |
| `torch.int8` / `torch.uint8` | 1 | Quantized inference. |
| `torch.bool` | 1 | Masks. `(x > 0)` returns bool. |

```python
# convert dtypes
t.float()        # to float32
t.long()         # to int64
t.bool()         # to bool
t.to(torch.float16)  # explicit
```

### Common dtype gotchas

```python
torch.tensor([1, 2])        # int64 (because the literals are int)
torch.tensor([1., 2.])      # float32 (because of the .)
torch.tensor([1, 2.])       # float32 (mixed → promoted)

# this fails: nn.Linear weights are float32 by default,
# you can't multiply by an int64 tensor
x = torch.tensor([1, 2, 3])              # int64
lin = torch.nn.Linear(3, 1)
# lin(x.float())  ← OK
# lin(x)          ← RuntimeError
```

> 🔑 **Rule of thumb:** weights and activations are float (32 or 16); token IDs, class labels, and indices are long.

---

## 3. Devices — CPU and GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.zeros(3, device=device)              # create directly on device (best)
y = torch.ones(3).to(device)                   # create on CPU, move (small overhead)
z = torch.ones(3).cuda()                       # shorthand for to("cuda")
back_to_cpu = z.cpu()                          # back

# query
torch.cuda.is_available()                      # True/False
torch.cuda.device_count()                      # # of GPUs
torch.cuda.current_device()                    # int
torch.cuda.get_device_name(0)
```

Two tensors must be on the **same device** for any binary op:

```python
a = torch.zeros(3, device="cuda")
b = torch.zeros(3, device="cpu")
a + b   # RuntimeError: Expected all tensors to be on the same device
```

> 🔑 **Pattern:** at the start of training, pick a `device` once. `.to(device)` your model and every batch. Never let CPU/GPU mismatch happen in the loop.

---

## 4. Indexing & Slicing

PyTorch uses NumPy-style indexing. Examples on a `(2, 3, 4)` tensor:

```python
x = torch.arange(24).reshape(2, 3, 4)

x[0]                # shape (3, 4) — first batch
x[0, 1]             # shape (4,)   — first batch, second row
x[0, 1, 2]          # scalar — single element
x[:, 0, :]          # shape (2, 4) — first row of every batch
x[..., -1]          # shape (2, 3) — "..." = "all preceding dims"; last col
x[0, :, 1:3]        # shape (3, 2) — slice columns 1 and 2

# advanced indexing
x[[0, 1], [1, 2]]      # shape (2, 4) — picks x[0,1] and x[1,2]
mask = x > 10          # bool tensor of same shape
x[mask]                # 1D tensor of all elements > 10

# adding / removing dims
x[None]                # shape (1, 2, 3, 4) — new dim at front
x[:, None]             # shape (2, 1, 3, 4) — new dim at position 1
x.unsqueeze(0)         # equivalent to x[None]
x.squeeze()            # remove ALL size-1 dims
x.squeeze(0)           # remove dim 0 if size 1
```

### Setting values

```python
x[0] = 0                    # whole sub-tensor
x[..., 0] = 5               # all "first columns" → 5
x[mask] = 0                 # zero out where mask is True
x[x < 0] = 0                # replace negatives with 0 (= relu, kind of)
```

> ⚠️ Boolean indexing on the LHS *modifies in place*. If you also have `requires_grad=True` autograd may complain — use `torch.where` for differentiable replacement.

```python
torch.where(x > 0, x, torch.zeros_like(x))   # = relu(x), differentiable
```

---

## 5. Reshape vs. View vs. Permute vs. Transpose vs. Contiguous

This is **the** subject every PyTorch beginner gets wrong. Pay attention.

```python
x = torch.arange(24).reshape(2, 3, 4)

x.view(6, 4)                # change shape; requires contiguous memory; FREE (no copy)
x.reshape(6, 4)             # same shape change; copies if needed; safe
x.flatten()                 # → (24,)
x.flatten(start_dim=1)      # keep dim 0, flatten the rest → (2, 12)

x.transpose(0, 1)           # swap two specific dims — shape (3, 2, 4)
x.permute(2, 0, 1)          # full reorder by axis index — shape (4, 2, 3)
x.T                         # only for 2-D, transposes
```

### What "contiguous" means

Tensors in memory are stored as a flat 1-D buffer. The shape and **strides** describe how to walk that buffer. After `permute` or `transpose`, the tensor is the same data but with rearranged strides — its memory layout doesn't match its logical shape.

`view` requires contiguous memory; `reshape` does the right thing automatically (copies if needed). The classic crash:

```python
x = torch.randn(2, 3, 4)
y = x.permute(0, 2, 1)        # logically (2, 4, 3), but non-contiguous
y.view(2, 12)                  # ← CRASHES
y.contiguous().view(2, 12)     # ← works (forces a copy first)
y.reshape(2, 12)               # ← also works (does the copy for you)
```

> 🐛 **The most common transformer bug.** After `q.transpose(1, 2)` to move heads ahead of time, you can't `.view` it. Either use `.reshape()` or always call `.contiguous().view(...)`. Memorize this pattern.

### When to use which

| Tool | Use when |
|---|---|
| `view` | You know the tensor is contiguous and want a free reshape. |
| `reshape` | You just want it to work. (slightly slower if a copy happens) |
| `transpose(a, b)` | Swap exactly two dims. |
| `permute(*dims)` | Arbitrary reorder — supply all dims. |
| `flatten` / `unflatten` | Collapse or expand a contiguous range of dims. |
| `unsqueeze` / `squeeze` | Add or remove size-1 dims. |
| `contiguous()` | Force a memory copy so future `view`s work. |

---

## 6. Combining and Splitting

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)

torch.cat([a, b], dim=0)     # (4, 3) — concatenate along an EXISTING dim
torch.cat([a, b], dim=1)     # (2, 6)
torch.stack([a, b], dim=0)   # (2, 2, 3) — adds a NEW dim at front
torch.stack([a, b], dim=-1)  # (2, 3, 2) — adds a NEW dim at end

# splitting
big = torch.randn(10, 4)
big.split(3, dim=0)          # tuple of 4 tensors, sizes (3, 3, 3, 1)
big.chunk(3, dim=0)          # tuple of 3 tensors, ~equal sizes
torch.unbind(big, dim=0)     # tuple of 10 tensors of shape (4,)
```

`cat` keeps dimension count the same. `stack` adds one. Mixing them up is the second-most-common shape bug.

---

## 7. Broadcasting (the rules)

Broadcasting lets you do ops between different-shape tensors without manually expanding. It's everywhere in DL.

**Rules** (compare shapes right-to-left):

1. Two dims are compatible if they're **equal**, or **one is 1**, or **one is missing**.
2. Size-1 dims are virtually replicated to match.
3. Missing dims are treated as size 1 prepended.

### Examples

```python
x = torch.randn(4, 1)
y = torch.randn(1, 3)
(x + y).shape                      # (4, 3) — broadcast both ways

x = torch.randn(B=8, T=10, d=32)
bias = torch.randn(d)
(x + bias).shape                   # (8, 10, 32) — bias broadcasts over B, T

q = torch.randn(B, h, T, d_h)
k = torch.randn(B, h, T, d_h)
scores = q @ k.transpose(-2, -1)   # (B, h, T, T) — matmul on last 2 dims, broadcasts over (B, h)

mask = torch.tril(torch.ones(T, T))            # (T, T)
scores.masked_fill(mask == 0, -float("inf"))   # broadcasts to (B, h, T, T)
```

### When broadcasting goes wrong

```python
a = torch.randn(3, 4)
b = torch.randn(4, 3)
a + b   # RuntimeError — neither dim matches and neither is 1
```

The fix is usually a `transpose`, `unsqueeze`, or `view` to align the shapes.

> 🧠 **Read shapes right-to-left.** Always. Walking from the rightmost dim makes broadcasting compatibility immediately obvious.

---

## 8. Math Operations

### Elementwise

```python
a + b, a - b, a * b, a / b, a % b, a ** b
torch.exp(a), torch.log(a), torch.log1p(a)
torch.sin(a), torch.cos(a), torch.tan(a)
torch.sqrt(a), torch.rsqrt(a)             # 1/sqrt
torch.abs(a), a.abs()                     # methods exist for most ops
torch.sign(a)
a.clamp(min=0, max=1)                     # clip values
torch.minimum(a, b), torch.maximum(a, b)  # elementwise
a.round(), a.floor(), a.ceil()
```

### Reductions (collapse one or more dims)

```python
a.sum()                       # scalar
a.sum(dim=-1)                 # collapse last dim
a.sum(dim=-1, keepdim=True)   # keep it as size 1 (handy for broadcasting back)
a.mean(dim=0)
a.var(dim=-1, unbiased=False)
a.std(dim=-1)

a.max(dim=-1)                 # returns NAMED TUPLE (values, indices)
a.amax(dim=-1)                # just values
a.argmax(dim=-1)              # just indices
a.min(dim=-1)                 # also (values, indices)

a.norm(dim=-1)                # Euclidean norm
a.norm(p=1, dim=-1)           # L1 norm
```

> ⚠️ `a.max(dim=-1)` returns a **named tuple**, not a tensor. `m = a.max(dim=-1); m + 1` fails confusingly. Either unpack `vals, idxs = a.max(dim=-1)` or use `a.amax(dim=-1)`.

### Matrix / batched matrix multiplication

```python
a @ b                         # operator form — preferred
a.matmul(b)                   # equivalent
torch.bmm(a, b)               # explicit batched: (B, n, m) @ (B, m, p) → (B, n, p)

# batched matmul auto-broadcasts via @ on tensors with >= 2 dims:
a = torch.randn(B, h, T, d)
b = torch.randn(B, h, d, T)
(a @ b).shape                 # (B, h, T, T)
```

### Einsum — the escape hatch

When matmul shapes get hairy, einsum is the clearest, most explicit option:

```python
# attention scores via einsum
# q: (B, h, T_q, d), k: (B, h, T_k, d) → scores: (B, h, T_q, T_k)
scores = torch.einsum("bhtd,bhsd->bhts", q, k)

# weighted value: scores (B, h, T, T) × v (B, h, T, d) → out (B, h, T, d)
out = torch.einsum("bhts,bhsd->bhtd", attn, v)
```

Same speed as the equivalent matmul, but the index notation makes the contraction explicit. **Worth learning** — saves real debugging time on multi-axis ops.

### Softmax & log-softmax

```python
import torch.nn.functional as F

probs = F.softmax(logits, dim=-1)
log_probs = F.log_softmax(logits, dim=-1)   # numerically stable log(softmax(x))
```

Always specify the `dim` you want to *normalize over*. For attention, that's the keys dim. For classification, it's the class/vocab dim.

---

## 9. In-Place Operations

Methods ending in `_` modify the tensor in place:

```python
x = torch.zeros(3)
x.add_(1)             # x is now [1, 1, 1]
x.zero_()             # in-place zero
x.fill_(7)            # in-place fill
x.normal_(mean=0, std=0.02)   # in-place random fill
x.uniform_(-1, 1)
```

In-place saves memory but can break autograd if a needed activation gets overwritten. **Default to non-in-place.** Use in-place only when you've measured that memory matters.

---

## 10. Summary Cheat Sheet

```python
# create
torch.tensor(data) | zeros, ones, full, empty | rand, randn, randint
arange, linspace, eye | *_like

# inspect
.shape .dtype .device .numel() .ndim .requires_grad

# device
.to(device) .cuda() .cpu()

# index
[i, j, k] | [..., -1] | [None] | unsqueeze | squeeze
boolean masks | torch.where

# reshape
view (free, contiguous req'd) | reshape (safe) | flatten/unflatten
permute (reorder dims) | transpose (swap 2 dims) | contiguous (force copy)

# combine
torch.cat (existing dim) | torch.stack (new dim)
.split() .chunk() .unbind()

# broadcast
right-to-left rule | size 1 = virtual replicate

# math
elementwise: + - * / | exp log sqrt abs clamp
reductions: sum mean max amax argmax norm (with dim, keepdim)
matmul: @ bmm einsum
```

📖 Next deep-dive: [02_autograd.md](02_autograd.md) — how gradients actually flow.
