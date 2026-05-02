# RLHF (PPO) vs GRPO — Why DeepSeek killed the critic

## Classic RLHF pipeline (InstructGPT, ChatGPT through ~2024)

```
1. SFT model from human demos.
2. Train a REWARD MODEL on human preference pairs (A vs B).
3. PPO: policy + value(critic) network, optimize policy w.r.t. reward model + KL anchor.
```

Memory in step 3: you have **4 models in GPU at once**:
- the trainable policy
- the frozen reference policy (for KL)
- the reward model
- the value/critic network

For a 7B policy, you need ~4× the memory of inference. Ugly.

## GRPO — drop the critic

For each prompt, sample $G$ outputs from the current policy. Compute reward for each. **Use the group's mean and std as your baseline**, not a learned critic.

$$A_i = \frac{r(o_i) - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

Then standard PPO-clipped policy gradient, plus KL to a reference model:

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_i \min(\rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)\right] - \beta\, \text{KL}(\pi_\theta \| \pi_\text{ref})$$

That's it. No value head, no critic loss, no second optimizer.

## DeepSeek-R1's other big idea: rule-based rewards

For verifiable domains (math, code, structured output), you don't need a learned reward model at all. Just:

```python
def reward(response, ground_truth):
    answer = extract_answer_tag(response)
    return 1.0 if answer == ground_truth else 0.0
```

This is gameproof — you literally can't reward-hack "is the math correct" because checking is automatic.

For non-verifiable stuff (helpfulness, tone), you still need preference data. R1 uses both: rules for math/code, learned RM for general behaviors in later stages.

## Why this matters

- **Memory:** GRPO uses ~half the VRAM of PPO+critic. Suddenly fine-tuning a 70B with RL is plausible on smaller clusters.
- **Stability:** the group baseline is variance-reduction without learned hyperparameters.
- **Training compute:** smaller, simpler.
- **Reasoning emerges from rule-based rewards.** This is the genuinely surprising scientific finding — you don't need to teach a model to reason via human-labeled chains; you just reward correct final answers and let RL figure out CoT structure on its own.

## See also

- Schulman et al., *PPO*, 2017.
- Ouyang et al., *InstructGPT*, 2022 — the canonical RLHF reference.
- Shao et al., *DeepSeekMath / GRPO*, 2024.
- DeepSeek-AI, *DeepSeek-R1*, 2025 (the paper in your folder).
