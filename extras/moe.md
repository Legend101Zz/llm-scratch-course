# Mixture of Experts (MoE)

> Used in DeepSeek-V3 (and therefore R1), Mixtral, GPT-4 (rumored). The way to scale parameters without proportionally scaling compute.

## The idea

Replace the FFN of each transformer block with $N$ parallel "expert" FFNs and a tiny **router** network. For each token, the router picks the **top-k** experts (typically k=1 or 2). Only those experts run for that token. Other experts: zero compute for that token.

```
Standard FFN:        x ──[FFN]──→ y      compute: 1 FFN's worth
MoE (top-2, N=8):    x ──[router]──→ {expert_3, expert_5}
                      └──[E_3]──┐
                                ├──→ y
                      └──[E_5]──┘     compute: 2 FFNs out of 8 active

Params total:  N × FFN_params (e.g. 8×)
Active params: k × FFN_params (e.g. 2×)
```

So a 671B-parameter MoE model (DeepSeek-V3) only activates ~37B params per token. You get the **knowledge** of the big model, the **compute** of the small one.

## Routing math

```python
# logits per expert
gate = x @ W_gate                       # (B, T, N)
top_k_vals, top_k_idx = gate.topk(k=2, dim=-1)
weights = softmax(top_k_vals, dim=-1)   # normalize to sum to 1

y = 0
for slot in range(k):
    expert_id = top_k_idx[..., slot]
    out = experts[expert_id](x)
    y = y + weights[..., slot:slot+1] * out
```

In practice this is fused and batched intelligently because naive token-by-expert scattering is slow.

## The hard parts

1. **Load balancing.** Without intervention, the router collapses to "use only expert 3 for everything". Solved with an **auxiliary loss** that penalizes uneven expert usage. (DeepSeek-V3 uses an auxiliary-loss-free strategy with bias updates.)
2. **Expert capacity.** Each expert can only handle so many tokens per batch — overflow tokens get dropped or routed to a fallback.
3. **Distributed training.** Experts often live on different GPUs (expert parallelism). Routing means tokens fly between GPUs every layer — the all-to-all communication is the bottleneck.

## Why R1 is built on a MoE base

- Cheap inference (37B active out of 671B total).
- Massive parameter count = lots of "knowledge slots" for the model to specialize.
- The reasoning training stays affordable because you're sampling at active-param scale, not total-param scale.

## See also

- Shazeer et al., *Outrageously Large Neural Networks (Sparsely-Gated MoE)*, 2017 (the original).
- *Switch Transformer*, 2021 (top-1 routing).
- *Mixtral 8x7B*, 2023 (open-source 8 experts, top-2).
- *DeepSeek-V3*, 2024 (auxiliary-loss-free balancing, 256 experts).
