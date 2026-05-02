# 🌌 Extras — modern transformer ideas

Read AFTER the main course. These are the differences between a 2017 transformer and a 2026 production model. Each topic is one short page.

| Topic | Why it matters |
|---|---|
| [`rope.md`](rope.md) | Modern position encoding (LLaMA, DeepSeek, Qwen all use it) |
| [`gqa_mqa.md`](gqa_mqa.md) | Smaller KV cache → longer context, faster inference |
| [`flash_attention.md`](flash_attention.md) | The kernel that runs every modern LLM |
| [`moe.md`](moe.md) | Mixture of Experts — DeepSeek-V3 / Mixtral architecture |
| [`scaling_laws.md`](scaling_laws.md) | Chinchilla / Kaplan — how to spend compute |
| [`rlhf_vs_grpo.md`](rlhf_vs_grpo.md) | Why GRPO replaced classic PPO-RLHF for reasoning |
| [`quantization.md`](quantization.md) | int8 / fp8 / nf4 — how to fit big models on small GPUs |
