# 6.3 — `nn.Module` and the Building Blocks

> Goal: know every `nn.*` block you'll meet in real models — what it does, what its shapes are, how its parameters are stored, and what its quirks are.

---

## 1. What `nn.Module` Is

`nn.Module` is a **container that owns parameters and a `forward` method**. Anything you assign to `self.xxx` that is itself a `nn.Module` or an `nn.Parameter` gets automatically registered. That registration is what makes:

- `model.parameters()` return all weights/biases recursively
- `model.to(device)` move all sub-modules
- `model.state_dict()` capture every weight for saving
- `model.train()` / `model.eval()` propagate to every sub-module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                          # MUST call this first
        self.l1 = nn.Linear(d_in, d_hidden)         # auto-registered
        self.l2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.dropout(x)
        return self.l2(x)

model = MLP(2, 16, 1)
print(model)                                        # nice tree printout
print(sum(p.numel() for p in model.parameters()))   # total param count
```

> 🔑 **Forgetting `super().__init__()`** is a common bug. Without it, registration breaks — your sub-modules become invisible. Always call it.

---

## 2. `nn.Parameter` — How Tensors Become "Weights"

A `nn.Parameter` is a tensor subclass that, when assigned to a `Module`, automatically gets `requires_grad=True` and registered.

```python
class MyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))    # learnable
        self.bias  = nn.Parameter(torch.zeros(d))   # learnable

    def forward(self, x):
        return x * self.scale + self.bias
```

You usually don't write `nn.Parameter` by hand — it's wrapped inside the standard layers. But it's essential when you build a custom layer.

### `register_buffer` — non-learnable persistent state

Some tensors are part of the module but **not learnable** — like a causal mask, position table, or running BatchNorm statistics. Register them as buffers.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    def forward(self, x, scores):
        T = scores.size(-1)
        return scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
```

Buffers:
- Move with `.to(device)` like parameters.
- Are saved/loaded with `state_dict()`.
- Do **not** appear in `.parameters()` and don't get gradients.

> 🔑 If you write `self.mask = torch.tril(...)` (no `register_buffer`), the mask **stays on CPU** when you do `model.to("cuda")`. Cryptic crashes follow.

---

## 3. The Building Blocks (alphabetical reference)

### `nn.Linear(in_features, out_features, bias=True)`

The fundamental `y = x W^T + b` layer. The single most-used module in PyTorch.

```python
lin = nn.Linear(8, 16)
lin.weight.shape            # (16, 8)  ← out × in
lin.bias.shape              # (16,)
y = lin(x)                  # x: (..., 8) → y: (..., 16)
```

Note the weight is stored as `(out, in)` — opposite of what you might expect. PyTorch does `x @ W.T + b` internally.

Trainable params: `in*out + out` (or `in*out` with `bias=False`).

### `nn.Embedding(num_embeddings, embedding_dim)`

A learnable lookup table. Maps integer IDs → vectors. Used for token embeddings, position embeddings.

```python
emb = nn.Embedding(vocab_size=50257, embedding_dim=768)
ids = torch.tensor([[1, 2, 3], [4, 5, 6]])    # (B=2, T=3) of int64
out = emb(ids)                                # (2, 3, 768)
```

Internally just a `(num_embeddings, embedding_dim)` weight matrix indexed by `ids`. The "lookup" is a differentiable index.

> ⚠️ Input must be `int64` (`.long()`). Float inputs error out.

### `nn.LayerNorm(normalized_shape)`

Per-token normalization (across features). Standard in transformers.

```python
ln = nn.LayerNorm(768)               # normalizes over last dim of size 768
y = ln(x)                            # x: (B, T, 768) → y same shape
ln.weight.shape                      # (768,) — gamma
ln.bias.shape                        # (768,) — beta
```

Computes `(x - mean) / sqrt(var + eps) * gamma + beta`, where mean/var are computed *per-token* over the feature dim.

### `nn.BatchNorm1d` / `BatchNorm2d` / `BatchNorm3d`

Normalizes across the **batch** dim. Very common in CNNs; almost never used in modern transformers.

Tracks running mean/var as buffers — different behavior in `train()` vs `eval()`. This is one big reason transformers prefer LayerNorm — no train/eval state divergence.

### `nn.Dropout(p)`

Zero each entry independently with probability `p` during training. Auto-disables under `eval()`.

```python
drop = nn.Dropout(0.1)
m.train(); drop(x)        # ~10% zeroed, rest scaled by 1/(1-p)
m.eval();  drop(x)        # identity, returns x
```

Always rely on `model.eval()` to turn it off — don't try to disable dropout by setting `p=0`.

### `nn.GELU()` / `nn.ReLU()` / `nn.SiLU()` / `nn.Tanh()`

Stateless activations. Same effect as the functional version; just packaged as a `Module` so it shows up in `print(model)`.

```python
relu = nn.ReLU()
y = relu(x)             # = torch.relu(x), same thing
```

### `nn.Sequential(*modules)`

Runs modules in order. Quick stacking when there's no branching.

```python
mlp = nn.Sequential(
    nn.Linear(8, 16),
    nn.GELU(),
    nn.Linear(16, 8),
)
y = mlp(x)
```

You can name them too:

```python
nn.Sequential(OrderedDict([
    ("l1", nn.Linear(8, 16)),
    ("act", nn.GELU()),
    ("l2", nn.Linear(16, 8)),
]))
```

### `nn.ModuleList([...])` and `nn.ModuleDict({...})`

For programmatically created sub-modules (e.g., a stack of N transformer blocks). **Never use a plain Python list/dict** — sub-modules in plain containers are NOT registered.

```python
class GPT(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

This is how you build "for N layers, do thing" architectures.

### `nn.Conv1d` / `Conv2d` / `Conv3d`

Convolutions. Beyond our LM scope but the API is similar:

```python
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# input:  (B, 3, H, W)
# output: (B, 64, H, W)  (with padding=1 to preserve size)
```

### `nn.MultiheadAttention`

PyTorch's built-in attention. Fast, but the API is awkward (it expects shape `(T, B, d)` by default — yes, time first). For learning purposes we wrote our own in Modules 4-5; in production code you'd use `F.scaled_dot_product_attention` directly.

```python
import torch.nn.functional as F
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# q, k, v: (B, n_head, T, d_head); y: same shape
```

This auto-picks FlashAttention or memory-efficient attention based on hardware. **Use this in real code.**

### `nn.RNN`, `nn.LSTM`, `nn.GRU`

Recurrent layers. Less relevant in 2026 but you'll see them in older models.

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
out, (h, c) = lstm(x)        # x: (B, T, 128) when batch_first=True
```

`batch_first=True` is non-default but you'll always want it — without it the input is `(T, B, ...)` which is confusing.

---

## 4. Iterating Over a Module

Three methods to know:

```python
model.parameters()              # iterator over all leaf tensors that need grad
model.named_parameters()        # → (name, tensor) pairs — useful for filtering
model.named_modules()           # → (name, module) pairs — every nested module
model.named_buffers()           # → (name, tensor) for non-learnable state

for name, p in model.named_parameters():
    print(name, p.shape)

# common pattern: separate weight decay groups
decay, no_decay = [], []
for name, p in model.named_parameters():
    if "bias" in name or "ln" in name or "embedding" in name:
        no_decay.append(p)
    else:
        decay.append(p)

opt = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

This pattern (no decay for biases/norms/embeddings) is standard in transformer training.

### Counting parameters

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{count_params(model)/1e6:.2f}M params")
```

---

## 5. Initialization

PyTorch picks reasonable defaults, but for transformers you usually override:

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

model.apply(init_weights)             # walks every sub-module
```

Common init functions:

```python
nn.init.zeros_(t)
nn.init.ones_(t)
nn.init.normal_(t, mean, std)
nn.init.uniform_(t, a, b)
nn.init.xavier_uniform_(t)            # 1/sqrt(fan_in) — Glorot
nn.init.kaiming_normal_(t)            # for ReLU networks — He init
```

GPT-2 uses `normal_(std=0.02)` for most weights; for residual-projection weights a smaller std `0.02 / sqrt(2 * n_layer)` to keep activations stable through the stack.

---

## 6. `train()` vs `eval()`

```python
model.train()        # dropout active, batchnorm uses batch stats
model.eval()         # dropout off, batchnorm uses running stats
```

Two layers care about this: `Dropout` and `*BatchNorm`. Everything else ignores it. **Forgetting `model.eval()` during evaluation is a notorious bug** — you'll get worse-than-expected metrics from dropout firing.

The flag propagates recursively. Setting it on the top-level module sets it on every sub-module.

---

## 7. Saving & Loading

**Save the `state_dict`, not the model object.**

```python
torch.save(model.state_dict(), "ckpt.pt")
```

Loading:

```python
model = MLP(2, 16, 1)                      # rebuild architecture
sd = torch.load("ckpt.pt", map_location="cpu")
model.load_state_dict(sd)
```

`map_location` is important when loading a CUDA-saved checkpoint on a CPU-only machine.

### Loading partial state

Sometimes you want to load weights for *most* keys (e.g., reusing a pretrained backbone with a new head):

```python
model.load_state_dict(sd, strict=False)
```

`strict=False` reports missing/unexpected keys but doesn't crash.

### Saving more than weights

A real training checkpoint also includes the optimizer state, LR schedule, and step counter:

```python
torch.save({
    "model": model.state_dict(),
    "opt": opt.state_dict(),
    "step": step,
    "rng_state": torch.get_rng_state(),
}, "ckpt.pt")
```

---

## 8. Functional API (`torch.nn.functional` / `F`)

Many `nn.*` modules have stateless `F.*` equivalents:

```python
import torch.nn.functional as F

F.linear(x, W, b)             # = nn.Linear(...).forward(x), but you supply W, b
F.relu(x), F.gelu(x), F.silu(x)
F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)
F.dropout(x, p=0.1, training=model.training)
F.layer_norm(x, normalized_shape, weight, bias)
F.cross_entropy(logits, targets)
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Use the **module form** when there's learned state (Linear, LayerNorm, Embedding). Use the **functional form** for stateless ops (activation, softmax, masking) — keeps code shorter and more readable.

---

## 9. Common Patterns Cheat Sheet

### "Stack of N blocks"

```python
class GPT(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return x
```

### "Add learnable scale on top of an op"

```python
class ScaledOut(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))   # learned per-feature scale
    def forward(self, x):
        return x * self.scale
```

### "Tie weights between two layers"

```python
self.lm_head.weight = self.token_emb.weight   # GPT-style weight tying
```

This shares the same tensor between the input embedding and the output projection — saves a lot of params and is empirically helpful for LMs.

### "Freeze a sub-module"

```python
for p in model.encoder.parameters():
    p.requires_grad = False
```

Common in fine-tuning (e.g., LoRA freezes the base model).

---

## 10. Reflection

1. Why does `nn.Parameter` exist instead of just using `torch.tensor(..., requires_grad=True)`?
2. You assign `self.layers = [Block() for _ in range(N)]` (plain list). What goes wrong? How does `nn.ModuleList` fix it?
3. When does `model.eval()` change behavior, and when is it a no-op?
4. Why save `state_dict()` rather than `torch.save(model, ...)`?
5. In weight tying (`lm_head.weight = token_emb.weight`), how does PyTorch know they're the *same* tensor (not just equal)? What does `.parameters()` count for this case?

📖 Next deep-dive: [04_optim_data_gpu.md](04_optim_data_gpu.md) — optimizers, data, devices, AMP.
