# 🐛 Debugging Cheatsheet — your future self will thank you

When (not if) something breaks, scan this list before despairing.

## SHAPE ERRORS — the #1 source of pain

### "RuntimeError: shape mismatch"

1. **`print(x.shape)` everywhere.** Sprinkle them through your forward pass. Remove them after.
2. **Convention:** `(B, T, C)` = batch, sequence, channels. Stick to it religiously.
3. After a `.view()` or `.reshape()`: did you account for *every* dim? `view` requires contiguous memory; if `transpose` was used, call `.contiguous()` first.
4. Multi-head attention: shape goes `(B, T, C) → (B, T, h, d_h) → (B, h, T, d_h)` (transpose 1↔2). Easy to mis-order.

### "expected dim N got dim M"

Usually a missing batch dim. Single example: `x.unsqueeze(0)` to add `B=1`.

## NaN / Inf in loss

Almost always one of:

1. **Learning rate too high.** First thing to try: divide by 10.
2. **No gradient clipping.** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
3. **Bad init.** `nn.init.normal_(p, std=0.02)` for transformers, not the default.
4. **`log(0)` somewhere.** Always `torch.log(p + 1e-12)`.
5. **Softmax over an all-`-inf` row.** Happens when the entire row of attention scores got masked. Off-by-one in the mask.
6. **Mixed precision overflow.** fp16 can overflow; switch to bf16 if T4 supports it (it doesn't — A100/H100 only). On T4 use fp16 with `GradScaler`.

```python
# detect NaN early:
torch.autograd.set_detect_anomaly(True)   # SLOW; only for debugging
```

## Loss not going down

In order:

1. **Did you `opt.zero_grad()`?** Forgetting this is the classic blunder.
2. **Are gradients nonzero?** `for n, p in model.named_parameters(): print(n, p.grad.norm() if p.grad is not None else 'None')`.
3. **Is `requires_grad=True` on your params?** `sum(p.requires_grad for p in model.parameters())`.
4. **Is the loss connected to params?** A `.detach()` or `.item()` somewhere kills the graph.
5. **LR too small** → stuck. Multiply by 10.
6. **LR too big** → loss bounces / NaN. Divide by 10.
7. **Mode mismatch:** Did you forget `model.train()` after `model.eval()`?

## Loss going down on train but val is flat / rising

Overfitting. Solutions in order:
1. More data (always #1).
2. Increase dropout (e.g. 0.1 → 0.3).
3. Add weight decay.
4. Smaller model.
5. Early stopping.

## Generation gibberish even after low loss

1. **Did you run `model.eval()`?** Dropout still on otherwise.
2. **Did you crop context to `block_size`?** Pos-embedding lookup will index out-of-range otherwise.
3. **Are you sampling, not greedy?** Greedy → loops. Use `temperature` and `top_k`.
4. **Tokenizer mismatch** — encoded with vocab A, decoded with vocab B. Check `stoi`/`itos` consistency.

## CUDA OOM (out of memory)

1. Reduce batch size first.
2. Enable `torch.cuda.amp.autocast()` (mixed precision).
3. Use `optimizer.zero_grad(set_to_none=True)` (frees grad tensors instead of zeroing).
4. Gradient checkpointing: `torch.utils.checkpoint.checkpoint(block, x)` — recomputes activations on backward, saves memory at cost of compute.
5. Smaller `block_size`.
6. LoRA instead of full fine-tune.

## Model loads but predicts nonsense after restart

Did you save AND load the optimizer state? `torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ...)`. Optimizer state matters for Adam (momentum buffers).

## Slow training (CPU-bound)

1. **Are you on GPU?** `next(model.parameters()).device` should be `cuda:0`.
2. **DataLoader workers:** `num_workers=4, pin_memory=True` for big datasets.
3. **CPU-side ops in the hot loop?** `print` is slow; remove from inner loop.
4. **`set_to_none=True` in `zero_grad`** — modest speedup.
5. **`torch.compile(model)`** in PyTorch 2.x — sometimes 1.5-2× free.

## "RuntimeError: leaf variable has been moved into the graph interior"

You did `param = param - lr * grad` directly. Use `with torch.no_grad(): param -= lr * grad` or use an optimizer.

## "Tensor on different device"

Move ALL inputs to the model's device:
```python
device = next(model.parameters()).device
x = x.to(device); y = y.to(device)
```

## DataLoader returns tensors of different sizes

Variable-length sequences. Either pad them in a `collate_fn` or set `drop_last=True` and chunk to fixed size.

## "Expected target size (...) got size (...)"

Cross-entropy expects:
- `logits`: `(N, V)` or `(B, V, T)` (for sequences)
- `targets`: `(N,)` or `(B, T)` — int64 class indices, NOT one-hot.

For LMs we usually flatten: `F.cross_entropy(logits.view(-1, V), targets.view(-1))`.

---

## Sanity checks to run on EVERY new model

```python
# 1) param count
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# 2) forward sanity
x = torch.zeros((1, 4), dtype=torch.long)
logits, loss = model(x, x)         # targets = inputs (dummy)
print(logits.shape)                 # (1, 4, V) expected
print(f"untrained loss: {loss.item():.4f}  (expected ~ln V = {math.log(V):.4f})")

# 3) overfit one batch test
# train ONLY on a batch of 4 examples for 1000 steps. Loss should go to ~0.
# If it doesn't, your model can't learn the task at all → architecture bug.

# 4) gradient flow
loss.backward()
for n, p in model.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        print(f"⚠️  zero grad: {n}")
```

The "overfit one batch" trick (#3) is the single best architecture sanity check. If it can't memorize 4 examples, no amount of data will save you.
