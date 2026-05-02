# 🗂️ Glossary — every term in this course, defined once

Cmd-F friendly. Open in a side pane while you work.

## Math / Calculus

- **Gradient** — vector of partial derivatives; the direction of steepest *increase* of a function. Negate it to descend.
- **Chain rule** — $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$. Backbone of backprop.
- **Jacobian** — matrix of all partial derivatives of a vector-valued function. We don't materialize it; backprop computes vector–Jacobian products implicitly.
- **Log-likelihood** — $\sum_i \log p(x_i)$. Maximizing this = minimizing negative log-likelihood = cross-entropy on one-hot targets.

## Autograd

- **Computation graph** — DAG of operations built during forward pass. Each node knows how to push gradient to its inputs.
- **Forward-mode AD** — propagate derivatives forward alongside values. Cheap when input dim is small. Useless for ML (millions of inputs).
- **Reverse-mode AD (backprop)** — propagate derivatives backward from the loss. Cheap when output dim is small (loss is scalar). What we use everywhere.
- **`requires_grad=True`** — tell PyTorch this tensor is a leaf parameter to track gradients for.
- **`.backward()`** — populates `.grad` on every leaf tensor in the graph.
- **Gradient accumulation (`+=`)** — required when a tensor feeds multiple downstream consumers.

## Neural net basics

- **Neuron** — `phi(w·x + b)` — weighted sum + bias + activation.
- **Layer** — collection of neurons computing in parallel.
- **MLP** — multi-layer perceptron; stack of fully connected layers.
- **Activation function** — non-linearity inserted between linear layers (otherwise the whole net collapses to a single linear map).
- **ReLU** — `max(0, x)`. Cheap, sparse, dead-neuron risk.
- **GELU** — `x · Φ(x)` (Φ is Gaussian CDF). Smoother than ReLU. Used in GPT-2/3.
- **Tanh** — `(e^x - e^-x)/(e^x + e^-x)`. Output in (-1, 1). Used in older nets, micrograd.
- **Softmax** — `exp(x) / sum(exp(x))`. Turns logits into a probability distribution.
- **Sigmoid** — `1/(1+e^-x)`. Output in (0,1). Almost never used in modern hidden layers.

## Loss functions

- **MSE (mean squared error)** — for regression.
- **Cross-entropy** — for classification / next-token prediction. `-sum(y_true · log(softmax(logits)))`.
- **NLL (negative log-likelihood)** — same as cross-entropy when targets are one-hot.

## Optimizers

- **SGD** — vanilla stochastic gradient descent. `θ -= lr * grad`.
- **Momentum** — running mean of past gradients smooths updates.
- **Adam** — per-parameter adaptive lr based on running estimates of mean (1st moment) and variance (2nd moment) of gradients.
- **AdamW** — Adam with **decoupled** weight decay (decay applied directly to weights, not folded into gradient). Standard for transformers.
- **Learning rate** — step size. Single most-tuned hyperparameter.
- **Warmup** — linearly grow lr from 0 to peak over first N steps. Stops early instabilities.
- **Cosine schedule** — decay lr smoothly to ~0 over training. Empirically good for LLMs.

## Regularization

- **Weight decay (L2)** — penalize large weights; shrink them each step.
- **Dropout** — randomly zero out activations during training; forces redundancy.
- **Gradient clipping** — cap gradient norm. Prevents single-bad-batch explosions.

## Tokenization

- **Token** — atomic unit the model sees. Could be a char, word, or subword.
- **Vocab size (V)** — number of distinct tokens.
- **Embedding** — learned vector per token. `(V, d_embd)` lookup table.
- **BPE (Byte-Pair Encoding)** — iteratively merge the most frequent adjacent pair to grow vocab. GPT-2 standard.
- **WordPiece / SentencePiece / Unigram** — alternative subword schemes.
- **OOV (out-of-vocab)** — token outside the vocab. BPE makes this rare; falls back to byte-level.

## Attention

- **Q (Query)** — "what I'm looking for".
- **K (Key)** — "what I offer".
- **V (Value)** — "what I'll contribute if attended to".
- **Scaled dot-product attention** — `softmax(QK^T / √d_k) V`.
- **Multi-head attention (MHA)** — parallel attention with different learned projections; concat outputs.
- **Causal / masked attention** — set future positions to `-inf` before softmax. Required for autoregressive LMs.
- **Self-attention** — Q, K, V all come from the same sequence.
- **Cross-attention** — Q from one sequence, K/V from another (encoder-decoder).
- **MQA (multi-query attention)** — share one K, V across all heads. Smaller KV cache.
- **GQA (grouped-query attention)** — middle ground; groups of heads share K, V. LLaMA-2/3 default.

## Transformer

- **Block** — one layer = LayerNorm → Attention → residual → LayerNorm → FFN → residual.
- **Pre-norm vs post-norm** — order of LN. Pre-norm trains more stably; modern standard.
- **FFN (feed-forward network)** — two-layer MLP inside a block. Width usually 4× embed dim.
- **Position embedding** — vector encoding "where in the sequence". Learned absolute (GPT-2), sinusoidal (Attention paper), or rotary (RoPE — modern).
- **RoPE (Rotary Position Embedding)** — rotate Q, K vectors by an angle proportional to position. Generalizes better, used in LLaMA/DeepSeek.
- **LayerNorm** — normalize across the feature dim per token. Learnable scale/shift.
- **RMSNorm** — LayerNorm without the mean-subtraction step. Cheaper. LLaMA default.
- **Residual connection** — `y = x + f(x)`. The gradient highway.
- **Weight tying** — share input embedding with output projection. Saves params, common in GPT-2.

## Training / inference

- **Pre-training** — next-token prediction on huge corpus.
- **SFT (supervised fine-tuning)** — fine-tune on (instruction, response) pairs.
- **RLHF** — reinforcement learning from human feedback; uses a learned reward model trained on human preferences.
- **PPO (Proximal Policy Optimization)** — RL algorithm with policy + value (critic) networks.
- **GRPO (Group Relative Policy Optimization)** — PPO without critic; advantage = (reward – group_mean)/group_std. DeepSeek's choice.
- **KL divergence** — distance between two probability distributions. Used as regularizer toward a reference policy.
- **KV cache** — at inference, cache K, V tensors of past tokens so each new token's attention doesn't reprocess them.
- **Speculative decoding** — small draft model proposes tokens; big model verifies in parallel. Faster inference.
- **Quantization** — store weights in 8-bit / 4-bit instead of 16-bit float. Less memory, slight quality loss.

## PEFT

- **LoRA (Low-Rank Adaptation)** — freeze W, add `BA` low-rank update. Few params trained.
- **QLoRA** — LoRA + 4-bit quantized base model. Fine-tune big models on small GPUs.
- **Adapter** — small bottleneck inserted between layers. Predates LoRA.

## Scaling

- **Scaling laws (Kaplan, Chinchilla)** — empirical: loss is a power law in compute, params, and data.
- **Chinchilla-optimal** — for a fixed compute budget, the right ratio is roughly **20 tokens per parameter**. (DeepMind, 2022.)
- **MoE (Mixture of Experts)** — replace FFN with N experts; a router picks k of them per token. Sparse compute, dense knowledge. DeepSeek-V3, Mixtral.

## Common acronyms

| Acronym | Meaning                                         |
|---------|-------------------------------------------------|
| LM      | Language Model                                  |
| LLM     | Large Language Model                            |
| CoT     | Chain of Thought                                |
| FFN     | Feed-Forward Network                            |
| FLOP    | FLoating-point OPeration                        |
| AMP     | Automatic Mixed Precision (fp16/bf16 + fp32)    |
| DDP     | Distributed Data Parallel                       |
| FSDP    | Fully Sharded Data Parallel                     |
| TBPTT   | Truncated Backprop Through Time (RNN-era)       |
| OOV     | Out Of Vocabulary                               |
| BOS/EOS | Beginning / End Of Sequence (special tokens)    |
| PAD     | Padding token (to align batch sequence lengths) |
