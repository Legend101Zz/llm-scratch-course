# 🚀 What's Next — after the 8-hour sprint

Congrats. You went from "I've watched videos about LLMs" to "I've built one". Here's the off-ramp into deeper waters, ranked by "bang for buck".

## Tier 1 — Solidify (do these next week)

1. **Re-do Modules 1, 4, 7 from scratch on a blank file.** No peeking. If you can't, you don't actually know it yet. Fixes superficial understanding.
2. **Watch Karpathy's full *Zero to Hero* playlist** — covers the same ground at a slower pace, with more on optimizers, batchnorm, and beyond. The lectures and your hands-on hours compound.
3. **Read the 3 papers in your folder, end-to-end this time** — now you have the vocabulary to actually parse them.
4. **Implement RoPE** (extras/rope.md) and swap it into your Module 7 GPT. Compare loss curves vs learned absolute embeddings on tinyshakespeare.

## Tier 2 — Scale up (the next month)

5. **nanoGPT on OpenWebText.** Karpathy's [nanoGPT repo](https://github.com/karpathy/nanoGPT) trains a real GPT-2-scale model. Rent an A100 for $1/hr on Lambda, train for a day, watch a 124M model emerge.
6. **Implement a KV cache** in your Module 7 generate function. Profile inference speedup.
7. **Implement GQA** in your attention module. Measure KV cache reduction.
8. **Build a real BPE tokenizer** (`bpe.py` in Module 3 was a toy — try matching `tiktoken` exactly on a small corpus).
9. **Read [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** with your own implementation open in another tab. Cross-reference.

## Tier 3 — Push the frontier (1-3 months)

10. **Implement GRPO end-to-end on GSM8K** (grade-school math). Use a small base model (Qwen 0.5B or DeepSeek-1.3B). Compare math accuracy before/after RL. This is the closest you'll come to reproducing R1.
11. **Implement Mixture of Experts** in your transformer block. Train on tinyshakespeare and watch routing patterns emerge.
12. **Read the FlashAttention paper** and implement the online softmax in pure CUDA / Triton. (Hard. Do this only if GPU kernels excite you.)
13. **Implement speculative decoding** — small draft model + your GPT as the verifier. Measure tokens/sec speedup.

## Tier 4 — Become dangerous (open-ended)

14. **Reproduce a recent paper** — pick from arxiv-sanity. The act of reproducing teaches more than 10 reads.
15. **Contribute to vLLM, llama.cpp, transformers, or trl.** Real production code, real reviewers, real growth.
16. **Train a domain-specific reasoner** — fine-tune a small base on a verifiable task you care about (chess puzzles? Codeforces? proof rewrites?). This is where the field is heading.

## Reading list (in priority order)

**Foundational:**
- Bishop, *Pattern Recognition and Machine Learning* — chapters on neural nets and backprop.
- Goodfellow, Bengio, Courville, *Deep Learning Book* — free online.

**Transformers and LLMs:**
- *The Illustrated Transformer* (Jay Alammar's blog).
- *Attention Is All You Need* — re-read once you've built one.
- *GPT-2 / GPT-3 / GPT-4* papers in chronological order.
- *Chinchilla*, *PaLM* (for scaling).
- *LLaMA* and *LLaMA-2* technical reports.

**RL and reasoning:**
- Schulman, *PPO*, 2017.
- Ouyang et al., *InstructGPT*, 2022.
- Shao et al., *DeepSeekMath* / GRPO, 2024.
- DeepSeek-AI, *DeepSeek-R1*, 2025 — the one in your folder.
- Anthropic's *Constitutional AI*.

**Mech interp (optional but mind-expanding):**
- Anthropic, *A Mathematical Framework for Transformer Circuits*.
- Anthropic, *Toy Models of Superposition*.
- The whole [transformer-circuits.pub](https://transformer-circuits.pub) corpus.

## Channels / blogs to follow

- Andrej Karpathy (YouTube).
- Sebastian Raschka (Substack — *Ahead of AI*).
- Jay Alammar (illustrated explanations).
- Lilian Weng (lilianweng.github.io — long-form deep dives).
- Simon Willison (simonwillison.net — daily news + experiments).
- Anthropic / OpenAI / DeepMind / DeepSeek release notes.

## Compute resources

- **Free:** Google Colab (T4), Kaggle (P100, 30 hrs/week).
- **Cheap:** Lambda Labs ($1/hr A100), RunPod (similar), vast.ai (cheaper, less reliable).
- **Frontier-ish:** TPUs via Google research credits, AWS/GCP/Azure if you have credits.

## Final advice

You learned more in 8 hours than most people who say they "study LLMs" learn in a month — because you typed every line. Keep that habit. Theory you can read; practice you have to bleed.

Now go build something stupid and ship it. 🥋
