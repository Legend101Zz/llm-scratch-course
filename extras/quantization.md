# Quantization — Big models on small GPUs

> The reason you can run a 70B model on a single 24GB consumer GPU.

## The problem

A 7B model in fp32 = 28 GB just for weights. fp16 halves it (14 GB). Most consumer GPUs can't fit a 70B model at fp16 (would need ~140 GB).

## The trick

Replace the 16-bit floats with 8-bit or 4-bit integers, with carefully designed scale/zero-point per group of weights so you lose minimal accuracy.

| Format | Size for 7B | Quality vs fp16 |
|---|---|---|
| fp32      | 28 GB | reference         |
| fp16/bf16 | 14 GB | identical         |
| int8      | 7 GB  | ~99% retained     |
| nf4 (QLoRA) | 3.5 GB | ~98% retained   |
| int2      | 1.75 GB | quality drops    |

## How it works (int8, "absmax" quantization)

For a tensor $W$:
1. Compute $s = \max(|W|) / 127$.
2. Store $W_q = \text{round}(W / s)$ as int8 in [-128, 127].
3. To use: $W \approx s \cdot W_q$ (cast to fp16 for matmul).

In practice you do this **per-row** or **per-block** of the weight tensor (better accuracy than per-tensor).

## Activations vs weights

- **Weight-only quantization:** weights int8/4, activations fp16. Easy, common (`bitsandbytes` `load_in_8bit`).
- **Full int8 (W8A8):** both weights and activations int8. Faster, but harder — outlier activations break naive scaling. Solutions: SmoothQuant, GPTQ, AWQ.

## QLoRA

Combines:
1. **4-bit quantized base model** (frozen, in nf4 — a non-uniform float-like 4-bit format).
2. **LoRA adapters** on top, in fp16.
3. **Paged optimizers** (offload optimizer states to CPU when not used).

Result: fine-tune a 65B model on a single 48GB GPU. Released in 2023, became the standard for community fine-tunes.

## Inference engines

- **`bitsandbytes`** — easy 8-bit and 4-bit loading. `from_pretrained(..., load_in_4bit=True)`.
- **GPTQ** — post-training quantization with ~4-bit and minimal quality loss. Used by AutoGPTQ, exllama.
- **AWQ** — Activation-aware Weight Quantization. Often beats GPTQ at lower bits.
- **GGUF / llama.cpp** — quantization formats for CPU/Apple Silicon inference.

## See also

- Dettmers et al., *LLM.int8()*, 2022.
- Frantar et al., *GPTQ*, 2022.
- Lin et al., *AWQ*, 2023.
- Dettmers et al., *QLoRA*, 2023.
