# 🥋 LLM-from-Scratch — 8 Hour Sprint Schedule

> See [`README.md`](README.md) for the project overview. This file is the detailed schedule + conceptual map.

> Goal: by the end of today you will have **(a)** built an autograd + tiny NN library by hand, **(b)** built a working GPT-2-style transformer in pure Python, **(c)** rebuilt it in PyTorch, **(d)** trained + fine-tuned it on Colab, and **(e)** understood the math + RL ideas behind DeepSeek-R1 well enough to implement a toy version.

This is a **sprint**. You will not finish everything perfectly. That's fine. The point is to *touch every concept with your fingers*, not to polish.

---

## 🧭 The Rules of This Dojo

1. **Type every line yourself.** Do NOT copy-paste my solutions until you've struggled. The struggle IS the lesson.
2. **Each module has 4 parts:** `README.md` (theory + math), `starter.py` (you fill TODOs), `solution.py` (revealed only after struggle), `notes.md` (your own scratch — write what confused you).
3. **Set a timer per module.** If you blow past 1.5× the estimate, peek at the solution, understand it, and move on. Sprint > perfection.
4. **Do the math by hand at least once per module.** A sheet of paper next to your keyboard. Non-negotiable.
5. **Talk to yourself out loud** when explaining tensor shapes. If you can't say it, you don't know it.

---

## 📅 The 8-Hour Schedule

| #  | Module                          | Time   | Output                                |
|----|---------------------------------|--------|---------------------------------------|
| 0  | Setup + mental model            | 15m    | env ready, you can draw an LLM        |
| 1  | Autograd engine (micrograd-ish) | 50m    | `Value` class with backprop           |
| 2  | Neural net on your autograd     | 30m    | MLP that learns XOR                   |
| 3  | Tokenization + bigram model     | 25m    | tokenizer + bigram baseline           |
| 4  | Self-attention from scratch     | 50m    | numpy QKV attention                   |
| 5  | Transformer block from scratch  | 40m    | full block in numpy                   |
| 6  | PyTorch crash course            | 30m    | rewrite Module 2 in torch             |
| 7  | Mini-GPT in PyTorch             | 60m    | nanoGPT-style model                   |
| 8  | Train on Colab                  | 45m    | a model that emits Shakespeare-y text |
| 9  | Fine-tuning + LoRA              | 30m    | LoRA-adapted model                    |
| 10 | DeepSeek-R1 / reasoning + GRPO  | 45m    | toy GRPO loop, reward shaping         |
|    | **Total**                       | **~7h 40m** | + 20m buffer for tea / panic    |

> 💡 If you fall behind: skip Module 5 (it's the most reusable from Module 4) and Module 9 (concept-only is fine). Never skip 1, 4, 7, 10.

---

## 🏗️ Conceptual Map — How It All Fits

```
                     ┌─────────────────────────────────┐
                     │  Calculus (chain rule)           │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 1 →    │  Autograd engine: scalar Value   │
                     │  + backward()                    │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 2 →    │  Neuron → Layer → MLP            │
                     │  trains on XOR via autograd      │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
   Modules 3-5 →     │  Tokens → embeddings →           │
                     │  attention → transformer block   │
                     │  (still pure python/numpy)       │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 6 →    │  Same things, but in PyTorch     │
                     │  (autograd is now free)          │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 7 →    │  Stack blocks → GPT-2 mini       │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 8 →    │  Pre-train on Colab (Tiny Shakes)│
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 9 →    │  Fine-tune + LoRA                │
                     └────────────┬────────────────────┘
                                  ▼
                     ┌─────────────────────────────────┐
       Module 10 →   │  RLHF/GRPO → reasoning models    │
                     │  (DeepSeek-R1)                   │
                     └─────────────────────────────────┘
```

---

## 📖 Always-on references (open in side panes)

- **[`glossary.md`](glossary.md)** — every term defined once. Cmd-F friendly.
- **[`debugging.md`](debugging.md)** — your first stop when something breaks. Shape errors, NaN losses, OOM, slow training.
- **[`extras/`](extras/)** — modern transformer ideas (RoPE, GQA, FlashAttention, MoE, scaling laws, GRPO vs RLHF, quantization). Read AFTER the main course.
- **[`whats_next.md`](whats_next.md)** — post-sprint progression. Tier 1 (consolidate) → Tier 4 (frontier).

## 📚 Papers (in [`papers/`](papers/))

- `1706.03762v7.pdf` — *Attention Is All You Need* (Vaswani et al., 2017). Read the abstract + Section 3 (model architecture) before Module 4.
- `language_models_are_unsupervised_multitask_learners.pdf` — *GPT-2* (Radford et al., 2019). Skim before Module 7.
- `2501.12948v2.pdf` — *DeepSeek-R1* (2025). Read before Module 10. Pay attention to GRPO + reward design.

See [`papers/README.md`](papers/README.md) for annotated cheat-sheets of each paper. Per-module pointers are in each module's README.

---

## 🛠️ Setup (do this now — 5 minutes)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll need a free Google Colab account for Module 8 onwards (GPU). [colab.research.google.com](https://colab.research.google.com) → New Notebook → Runtime → Change runtime type → T4 GPU.

---

## ▶️ Start Here

Open [`00_start/README.md`](00_start/README.md) and let's go. Clock starts when you do. 🕐
