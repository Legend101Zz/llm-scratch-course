# Scaling Laws — How to spend a million dollars on GPUs

> Two papers: Kaplan et al. 2020 (OpenAI) and Hoffmann et al. 2022 (DeepMind / *Chinchilla*).

## Kaplan (2020) — "Loss is a power law in compute"

Loss $L$ vs. compute $C$ (FLOPs):

$$L(C) \approx \left(\frac{C_c}{C}\right)^{0.05}$$

So 10× more compute → ~12% lower loss. The relationship was so clean for years that "just scale" was a viable AI strategy.

Kaplan also gave power laws in model size $N$ and dataset size $D$ separately. **Conclusion at the time:** for fixed compute, bigger models > more data.

## Chinchilla (2022) — Kaplan was wrong about the ratio

Hoffmann et al. retrained at hundreds of (N, D) combos and found the optimal allocation is roughly:

$$N_\text{optimal} \propto C^{0.5}, \quad D_\text{optimal} \propto C^{0.5}$$

i.e. you should **scale params and tokens equally**. As a rule of thumb, **20 tokens per parameter**.

So GPT-3 (175B params, 300B tokens) was *severely* under-trained. Chinchilla itself was 70B params trained on 1.4T tokens — and beat GPT-3 despite being half the size. *That's* the Chinchilla revolution.

## What this means in practice

- A 1B model wants ~20B tokens to be Chinchilla-optimal.
- A 70B model wants ~1.4T tokens. (LLaMA-2 70B trained on 2T — slightly over-trained, intentionally.)
- LLaMA-3 trained on 15T tokens for an 8B model. **Way past Chinchilla optimal.** Why? Because at deployment time you generate billions of tokens; an extra dollar at training pays back forever. Compute-optimal ≠ deployment-optimal.

## The "compute-optimal" formula (rough)

$$C \approx 6 N D \text{ FLOPs}$$

(Forward + backward at ~6× the param count per token.)

So if you have $10^{22}$ FLOPs of training budget, Chinchilla says:
- $N \approx 10^{10}$ (10B params)
- $D \approx 1.7 \times 10^{11}$ (170B tokens)

## Why this matters for us

- Don't bother training a tiny model on TBs of data — you're wasting both.
- Don't train a giant model on a small dataset — it'll just memorize.
- For a 1M-param toy on tinyshakespeare (1M chars): we are **massively under-data**, hence the model overfits easily.

## See also

- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022.
- Sardana et al., *Beyond Chinchilla-Optimal*, 2023 (when you should over-train).
