# Module 9 — Fine-tuning + LoRA (30 min)

> ⏱️ 30 min. The cheap trick that runs the whole "open-source fine-tune" industry.

## Two kinds of fine-tuning

### Full fine-tune
Same model, smaller LR (often 1/10th of pretraining), curated data, often shorter epochs. Update **all** parameters. Memory cost = same as pretraining. Expensive.

### Parameter-Efficient Fine-Tuning (PEFT) → LoRA
Update tiny additive matrices, freeze the rest. **0.1 – 1% of params** trainable. Memory drops 10×+ and you can fine-tune a 7B model on a single 16GB GPU.

## LoRA in one picture

For each linear layer with weight $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the forward pass is normally $y = Wx$.

LoRA replaces it with:

$$y = Wx + \frac{\alpha}{r} \, B A x$$

- $A \in \mathbb{R}^{r \times d_{in}}$, initialized random Gaussian.
- $B \in \mathbb{R}^{d_{out} \times r}$, initialized **zero**.
- $r$ is the **rank** — typically 4, 8, 16. Tiny.
- $\alpha$ is a scaling hyperparameter.

At init, $B = 0$ so the model is identical to the original. As training proceeds, $A, B$ learn a *low-rank update* to $W$. Number of new params: $r \cdot (d_{in} + d_{out})$ vs. $d_{in} \cdot d_{out}$ for full fine-tune. For $d=4096, r=8$: **64K vs 16M** — 250× fewer.

### Why low rank works

Empirically, the *change* a fine-tune induces in $W$ has surprisingly low intrinsic rank. The base model already has the right "feature directions"; fine-tuning mostly reweights them. Hu et al. (2021) showed $r=4$ often matches full fine-tune.

## Implementation (1 screen of code)

See [`lora.py`](lora.py). It defines a `LoRALinear` layer that wraps any `nn.Linear`. Usage:

```python
def replace_with_lora(module, r=8, alpha=16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            replace_with_lora(child, r, alpha)

# freeze all base params, only LoRA A,B are trainable:
for p in model.parameters(): p.requires_grad_(False)
replace_with_lora(model)
trainable = [p for p in model.parameters() if p.requires_grad]
```

## End-to-end training demo

Open [`lora_train_demo.py`](lora_train_demo.py). It does:
1. **Phase 1:** quick "pretrain" of a small GPT on lowercase tinyshakespeare (full fine-tune, 800 steps).
2. **Phase 2:** freeze all base params, install LoRA on every `nn.Linear`, fine-tune to **ALL CAPS** in 400 steps.

You'll see:
- "trainable: ~30k / total: ~700k (≈4%)" — LoRA's promise made concrete.
- Phase 2 loss drops fast (it's a small distribution shift).
- Final sample is uppercase Shakespeare-flavored text.

Run on Colab T4 — finishes in <2 min.

## Mention while we're here

- **QLoRA** = LoRA + 4-bit quantized base. Fits a 7B model on a single 16GB GPU. The library: `bitsandbytes`.
- **Where to apply LoRA:** Q, K, V, output proj, sometimes FFN. Bigger surface = more capacity but more params. `r=8, alpha=16` is a common starting point.
- **Inference cost:** zero. After training, `W' = W + (α/r) BA` and you ship one merged matrix. Original throughput.

## Reflection

- Why is $B$ initialized to zero, not $A$? (To start the residual at zero. If $A=0$, gradients to $A$ would also be zero — dead.)
- Where in a transformer does LoRA usually go? (Q, K, V projections; sometimes also FFN.)
- What's the inference cost of LoRA after training? (Zero — you can fold $BA$ into $W$.)

✅ Next: [Module 10 — DeepSeek R1 + GRPO](../10_deepseek_r1/README.md). The big finale.
