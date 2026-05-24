# 6.4 — Optimizers, Data Loading, GPU & Performance

> Goal: know how to actually train a model — pick the right optimizer, build a data pipeline, push everything to GPU correctly, and use mixed precision when needed.

---

## 1. Loss Functions

Use the **functional** form from `torch.nn.functional` (they're stateless and read more naturally):

```python
import torch.nn.functional as F

# regression
F.mse_loss(pred, target)
F.l1_loss(pred, target)            # mean absolute error
F.smooth_l1_loss(pred, target)     # Huber loss

# binary classification
F.binary_cross_entropy_with_logits(logits, target)
# logits: any shape; target: same shape, values in {0, 1}

# multiclass classification
F.cross_entropy(logits, target)
# logits: (N, C); target: (N,) of class indices in [0, C)
# also accepts target as a (N, C) probability distribution (label smoothing)

# negative log-likelihood (when you've already done log_softmax yourself)
F.nll_loss(log_probs, target)
```

> 🔑 **Use `*_with_logits` whenever it exists.** It folds `sigmoid`/`softmax` into the loss for numerical stability. Computing `softmax` then `log` then `nll` separately is a classic source of NaN.

### Cross-entropy for language models

```python
# logits: (B, T, V)   targets: (B, T) of token IDs
loss = F.cross_entropy(
    logits.reshape(-1, V),       # → (B*T, V)
    targets.reshape(-1),         # → (B*T,)
)
# with padding (ignore positions where target is the pad/ignore index)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
```

`ignore_index` is the standard convention — pad your target tensor with `-100` for positions you don't want to count toward the loss.

---

## 2. Optimizers

```python
import torch.optim as optim

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

### Which one?

- **SGD with momentum**: classic, still strong for vision (CNNs on ImageNet). Slow to tune.
- **Adam**: adaptive per-param learning rates. Robust default for everything in DL.
- **AdamW**: Adam + *decoupled* weight decay. **The default for transformers.** The L2 regularization in plain Adam interacts badly with the adaptive LR; AdamW fixes this.

For LLM training, betas of `(0.9, 0.95)` are common (slightly less aggressive than Adam's default `(0.9, 0.999)`).

### The training-step formula

```python
opt.zero_grad()       # clear .grad on every param
loss.backward()       # compute gradients
opt.step()            # update params using stored gradients
```

Forget any one and your model silently misbehaves. Memorize the order.

`opt.zero_grad(set_to_none=True)` is slightly faster than zeroing — sets `.grad` to `None` instead of overwriting with zeros. This is the default since PyTorch 2.0.

### Param groups (different LR or weight decay per param)

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if "bias" in n or "ln" in n or "norm" in n or "embedding" in n:
        no_decay.append(p)
    else:
        decay.append(p)

opt = optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

This pattern (no decay on biases/norms/embeddings) is standard for transformers.

---

## 3. Gradient Clipping (you'll need this for transformers)

Cap the global gradient norm to prevent occasional huge updates from blowing up training:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```

`max_norm=1.0` is standard for LLM training. This single line prevents most loss-explosion disasters.

> 🔑 Clip **after** `backward()` and **before** `step()`. Other order = no effect.

There's also `clip_grad_value_` (clip each entry individually); `clip_grad_norm_` is what you want 99% of the time.

---

## 4. Learning-Rate Schedulers

A constant LR rarely works. The standard transformer recipe is **linear warmup → cosine decay**:

```python
total_steps = 10_000
warmup_steps = 500

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))   # cosine from 1 to 0

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```

In the loop:

```python
for step in range(total_steps):
    ...
    opt.step()
    sched.step()                    # AFTER opt.step
```

### Built-in schedulers

```python
torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)        # decay 50% every 1k steps
torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000)
torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=10_000)   # 1cycle
torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3)     # adaptive
```

For LLM pre-training, `LambdaLR` with a custom warmup-then-cosine function (above) is most common.

---

## 5. Datasets and DataLoaders

PyTorch separates "where the data lives" (`Dataset`) from "how to iterate it efficiently" (`DataLoader`).

### `Dataset` — your data, indexed by integer

A custom dataset implements `__len__` and `__getitem__`:

```python
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = torch.tensor([ord(c) for c in text])
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y
```

Built-in datasets exist in `torch.utils.data` (`TensorDataset`, etc.) and `torchvision.datasets` / `torchtext.datasets` for vision/text.

### `DataLoader` — batching, shuffling, parallel I/O

```python
from torch.utils.data import DataLoader

dl = DataLoader(
    ds,
    batch_size=32,
    shuffle=True,           # reshuffle every epoch
    num_workers=4,          # parallel data loading processes
    pin_memory=True,        # speeds up CPU→GPU transfer for CUDA
    drop_last=True,         # drop the last incomplete batch
)

for x, y in dl:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    ...
```

Key knobs:

| Argument | Effect |
|---|---|
| `batch_size` | Items per batch. |
| `shuffle=True` | New random order each epoch (use only for training). |
| `num_workers` | Parallel CPU processes loading data. 0 = main process. |
| `pin_memory=True` | Allocates page-locked memory → faster H2D transfer on CUDA. |
| `drop_last=True` | Skip the final partial batch (avoids size mismatches). |
| `collate_fn` | Custom function to batch items (needed for variable-length data). |

> ⚡ For GPU training: `pin_memory=True`, `num_workers=2..8`, and pass `non_blocking=True` to `.to(device)`. These are easy 1.5–3× speedups on the data path.

### Variable-length batches (custom `collate_fn`)

If items have different lengths (typical in NLP), you write your own collate:

```python
def collate_fn(batch):
    xs = [item["input_ids"] for item in batch]
    lengths = [len(x) for x in xs]
    padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    return padded, torch.tensor(lengths)

dl = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
```

---

## 6. Devices and GPU Use

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# move model
model = model.to(device)

# move data inside the loop
for x, y in dl:
    x, y = x.to(device), y.to(device)
    ...
```

Rules:
- Both **model and data must be on the same device** for any op.
- Move the model **once** (outside the loop).
- Move data **every batch** (you can't keep the entire dataset in GPU RAM, usually).

### Multi-GPU (briefly)

For one machine with multiple GPUs, the modern approach is `torch.nn.parallel.DistributedDataParallel` (DDP) — one process per GPU, much faster than the older `DataParallel`. Out of scope here, but you should know:
- Single-GPU code: `model.to("cuda")`.
- Multi-GPU code: launch with `torchrun`, wrap with `DDP`, careful with all-reduce.

For our course, we stick to single-GPU (Colab T4).

---

## 7. Mixed Precision (AMP) — Free 2× Speedup on Modern GPUs

Mixed precision keeps weights in float32 but does the heavy compute in float16/bfloat16. **2–3× faster, half the memory, no quality loss** on transformers.

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")           # only needed for float16; bf16 doesn't need it

for x, y in dl:
    x, y = x.to(device), y.to(device)

    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
```

### bfloat16 vs float16

- **float16**: full speed, but limited range — losses can underflow. `GradScaler` handles this by scaling the loss up before backward and unscaling before optimizer.
- **bfloat16**: same range as float32, less precision. **No scaler needed.** Preferred on Ampere+ (A100, RTX 30xx, T4 supports it). Use this when available.

```python
with autocast(device_type="cuda", dtype=torch.bfloat16):    # no scaler at all
    logits = model(x)
    loss = ...

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
```

> 💡 Module 8 uses AMP — read its training script after this for a working example.

---

## 8. `torch.compile` — Automatic Speedup (PyTorch 2.x)

```python
model = torch.compile(model)         # one-line speedup
```

PyTorch 2.0+ introduces a JIT compiler that traces your model and generates fused, optimized kernels. Speedups vary: 10–100% for transformers, sometimes more.

Caveats:
- First batch is slow (compile pass).
- Some advanced patterns (heavy Python control flow, dynamic shapes) confuse it.
- Worth trying once your model trains correctly without it.

---

## 9. Profiling — Where Is My Time Going?

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        loss = model(x).sum()
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Quick win: print the top 20 ops by GPU time. If softmax dominates, you may benefit from `F.scaled_dot_product_attention`. If memory-copies dominate, your data path is bottlenecked.

For real profiling work, use `tensorboard` or `nsight-systems`.

---

## 10. The Canonical Training Loop

Putting it all together:

```python
import torch, torch.nn.functional as F
from torch.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
scaler = GradScaler("cuda")

model.train()
for step, (x, y) in enumerate(train_loader):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 100 == 0:
        print(f"step {step}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
```

Memorize this skeleton. **Every transformer training script in the wild looks like this.** Module 8's `train.py` is a real-world version.

### Evaluation block (drop into the loop periodically)

```python
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
```

`@torch.no_grad()` decorator + `model.eval()` — orthogonal flags, both needed.

---

## 11. Reflection

1. Why is **AdamW** (not Adam) the default for transformers?
2. What does `optimizer.zero_grad()` actually do? What goes wrong if you skip it?
3. Why does `pin_memory=True` speed up GPU training?
4. **bfloat16 vs float16** for AMP — when is each preferred?
5. You're on a Colab T4 (limited memory) and your model OOMs at batch size 32. List 5 ways to reduce memory without changing the model architecture.
6. After `torch.compile(model)`, the first batch is much slower than subsequent ones. Why?

---

## 12. What's NOT here (advanced topics)

- **DDP** (Distributed Data Parallel) — multi-GPU/multi-node training.
- **FSDP** (Fully Sharded Data Parallel) — train models bigger than one GPU's memory.
- **Gradient checkpointing** — trade compute for memory, see `extras/`.
- **Quantization** (int8 / int4 inference) — see `extras/07_quantization.md`.
- **Custom CUDA kernels** — for when fused ops aren't enough. Way out of scope.

These are post-course topics. The 11 sections above cover everything you need for Modules 7-10.

✅ Back to [Module 6 README](README.md). Or onward to [Module 7 — Mini-GPT](../07_gpt_pytorch/README.md).
