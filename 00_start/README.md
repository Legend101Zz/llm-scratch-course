# Module 0 — The Mental Model (15 min)

> ⏱️ Timer: **15 minutes**. We're not coding yet. We're calibrating.

## What is an LLM, in one paragraph?

An LLM is a function `f(context) → probability distribution over next token`. That's it. Everything else — attention, transformers, RLHF, "reasoning" — is engineering choices for how to *parameterize* that function and *train* it. Keep this north star: **next-token probability**.

```
"The cat sat on the"  →  f(·)  →  { "mat": 0.42, "floor": 0.18, "roof": 0.05, ... }
```

Sample from the distribution → append → repeat. That's "generation". An entire ChatGPT response is just this loop.

## What does training mean, mathematically?

Given a corpus, we want to maximize the probability the model assigns to the *actual* next token. Equivalently we minimize **negative log-likelihood** (a.k.a. cross-entropy):

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})$$

- $\theta$ = model parameters (the billions of numbers).
- $x_{<i}$ = all tokens before position $i$.
- $p_\theta$ = the probability the model assigns.

Training = use **gradient descent** on $\theta$ to push this loss down. To do gradient descent we need $\nabla_\theta \mathcal{L}$ — gradients. To get gradients automatically we need **autograd** (Module 1).

## Why three "stages" exist (pre-train → fine-tune → RL)

| Stage          | Data                             | Loss                       | What it teaches                      |
|----------------|----------------------------------|----------------------------|--------------------------------------|
| Pre-training   | Trillions of internet tokens     | Cross-entropy on next token| "What does language look like?"      |
| Fine-tuning    | Curated instruction/answer pairs | Cross-entropy (same!)      | "How to follow instructions"         |
| RL (RLHF/GRPO) | Prompts + reward signal          | Policy gradient            | "How to be helpful / reason / win"   |

DeepSeek-R1 (Module 10) is interesting because it shows you can do RL **directly for reasoning**, with rule-based rewards (math correct? code passes tests?), no human preference data needed.

## ✍️ Before continuing, on paper:

1. Draw the next-token-prediction loop with arrows.
2. Write the cross-entropy formula above without looking.
3. Answer: "If my model assigns probability 0.0001 to the correct next token, what is the loss for that token?" (Hint: $-\log(0.0001) \approx 9.21$.)

---

## Setup checklist

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
python -c "import torch; print(torch.__version__)"   # should print a version
```

Done? Move to **[Module 1 — Autograd Engine](../01_autograd/README.md)**.
