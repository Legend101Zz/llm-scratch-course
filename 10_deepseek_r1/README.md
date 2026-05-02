# Module 10 — DeepSeek-R1 & Reasoning Models via GRPO (45 min)

> ⏱️ 45 min. The finale. The thing that made everyone realize "RL on LLMs actually scales".
>
> 📄 Open `papers/2501.12948v2.pdf` (DeepSeek-R1). Read the abstract + Section 2 (Approach) + Section 2.2 (GRPO).

## What's the big idea?

Pre-trained LLMs already *know* a lot. They just don't always *use* that knowledge step-by-step. **Reasoning models** are LLMs trained to produce a long internal "chain of thought" before the final answer — and that chain dramatically improves performance on math, code, logic.

DeepSeek-R1 showed two surprising things:
1. **R1-Zero**: applying RL **directly on the base model** (no supervised fine-tuning first) with **rule-based rewards** (math: is the answer correct? code: do tests pass?) — produces strong reasoning behavior, including self-correction ("Wait, let me check..."), spontaneously.
2. The RL algorithm they used — **GRPO** — is much simpler than PPO and avoids needing a value/critic network.

## The Stack

```
   Pretrained base LM (DeepSeek-V3)
            │
            ▼  RL with rule-based rewards (math, code, format)
            │
   ┌───────────────────────┐
   │  R1-Zero              │   ← pure RL, no SFT — emergent reasoning
   └───────────────────────┘
            │
            ▼  + cold-start SFT data + multi-stage RL/refinement
            │
   ┌───────────────────────┐
   │  R1                   │   ← polished, deployable
   └───────────────────────┘
```

## GRPO — Group Relative Policy Optimization

Standard PPO has a policy + a value network (critic). The critic estimates "how good is this state?" and is used to compute advantages. GRPO **drops the critic entirely**.

**Key trick:** for each prompt $q$, sample a **group** of $G$ outputs $\{o_1, ..., o_G\}$ from the current policy. Score them all with a reward function $r(\cdot)$. Use the **group's own statistics** to normalize:

$$A_i = \frac{r(o_i) - \text{mean}(\{r(o_1), ..., r(o_G)\})}{\text{std}(\{r(o_1), ..., r(o_G)\})}$$

So $A_i$ is the per-output advantage relative to the group. Outputs better than average → positive advantage → push policy toward them.

The objective (PPO-clipped + KL-penalty form):

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\!\left(\rho_i A_i, \, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)\right] - \beta \, \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$$

where $\rho_i = \pi_\theta(o_i \mid q)/\pi_{\theta_\text{old}}(o_i \mid q)$ is the importance ratio, and $\pi_\text{ref}$ is the original SFT/base model (KL keeps us close to it so the model doesn't deteriorate).

### Reading this slowly

- $\rho_i$: how much MORE likely the new policy is to produce $o_i$, vs. the old policy. We *want* this to grow for high-advantage outputs.
- The `min(..., clip(...))` is PPO's hack: don't let importance ratios run away. Stable updates.
- $\text{KL}[\pi_\theta \| \pi_\text{ref}]$: anchor to the base model. Without it, RL can collapse into mode-degenerate junk.
- No value network → no critic loss → fewer hyperparams, smaller memory, simpler implementation.

### Reward Design (the unsexy part that matters most)

For R1-Zero on math:
- **Accuracy reward**: extract the final answer from `<answer>...</answer>` tags. If equal to ground truth → +1, else 0.
- **Format reward**: did the response include `<think>...</think><answer>...</answer>` correctly? +0.1 / 0.
- **No model-based reward** (no learned reward model). Pure rules.

This is wild because everyone assumed you needed a learned reward model (RLHF style). DeepSeek showed: for verifiable domains, rules are enough — **and they don't get reward-hacked**.

## A toy GRPO loop

Open [`grpo_toy.py`](grpo_toy.py). It implements GRPO on a **toy task**: train a policy to output the right digit when prompted with `"What digit is X?"`. The policy is a tiny model from Module 7. Reward is: answer == target → 1 else 0. The script:

1. Samples $G=8$ outputs per prompt.
2. Computes group-normalized advantages.
3. Updates the policy with clipped log-prob × advantage, plus KL to a frozen reference.

Won't make a real reasoner — but the **mechanics** are exactly the same as the paper. Once the toy loop trains, you understand R1.

## What they actually did differently from "vanilla GRPO"

- **Cold-start SFT data** (a few thousand high-quality CoT examples) before R1's main RL stage to reduce "language mixing" and weird formatting (R1-Zero produced legible-but-ugly chains).
- **Reasoning-oriented RL** (math, code) → second SFT stage with rejection sampling → general RL → final R1. Multi-stage.
- **Distillation** to smaller models: R1's outputs as SFT data for Qwen/Llama bases produces strong small reasoners.

## Reflection

- Why does dropping the critic save memory? (No second model the size of the policy.)
- The KL anchor uses $\pi_\text{ref}$, not the OLD policy from this step. Why? (We want long-horizon anchoring to base behavior, not short-horizon stability — which is what clipping handles.)
- Why does rule-based reward beat learned reward models for math/code? (Verifiable ground truth → can't be gamed by adversarial outputs.)

## Going further (if you have spare time)

- Read TRL's `GRPOTrainer` source — production GRPO in PyTorch.
- Try the toy script with $G=1$ vs $G=8$ — see what the variance reduction does.
- Add a length penalty to the reward and watch responses get shorter.

🥋 You finished the dojo. Now go pre-train something stupid on Colab and brag about it. Send the loss curve to a friend who doesn't know what a loss curve is.
