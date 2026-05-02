# Module 8 — Train On Colab (45 min)

> ⏱️ 45 min. We're using Colab's free T4 GPU (~16GB). On CPU this would take hours; on T4 it's ~5–10 min.

## Setup steps

1. Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. In a cell:
   ```python
   !nvidia-smi
   ```
   Confirm a Tesla T4 is shown.
4. Paste the contents of `train.py` (in this folder) into a single cell, run it. It will:
   - download tinyshakespeare,
   - build the char tokenizer,
   - instantiate the GPT from Module 7,
   - train for ~3000 steps with AdamW,
   - print loss + sample text every 500 steps,
   - save the final model.

Expected behavior:
- step 0:    loss ≈ 4.2 (random over 65 chars: $\ln 65 \approx 4.17$).
- step 500:  loss ≈ 2.5 (it's learning frequencies).
- step 1500: loss ≈ 2.0 (recognizable English-ish).
- step 3000: loss ≈ 1.5 (Shakespeare-flavored gibberish — *Cithens, my goodaught I should haves...*).

## What this teaches you

| Concept                       | Where it appears                                           |
|-------------------------------|------------------------------------------------------------|
| Mini-batch SGD                | `get_batch` samples random offsets each step               |
| Train/val split               | First 90% / last 10% of corpus                              |
| AdamW + weight decay          | `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)` |
| Learning rate warmup + cosine | Optional; we keep it simple here                            |
| Gradient clipping             | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`   |
| Mixed precision               | `torch.cuda.amp.autocast` (commented in the script)         |
| `model.train()` vs `.eval()`  | Toggles dropout                                              |
| Saving + loading              | `torch.save(model.state_dict(), "ckpt.pt")`                  |

## Why AdamW?

SGD with a constant lr on millions of params is hopeless — different params want different effective learning rates. Adam keeps a per-parameter running mean (1st moment) and variance (2nd moment) of gradients to *adaptively* scale steps. **AdamW** is Adam with weight decay applied directly to weights instead of folded into the gradient — this matters mathematically and empirically for transformers.

Update rule (sketch):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta \leftarrow \theta - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} - \eta \lambda \theta$$

## Reflection

- Why does loss start at $\ln(\text{vocab\_size})$?
- What's gradient clipping for and what bug does it prevent?
- If you ran for 100k steps on this tiny model, what happens? (overfit; train loss → 0, val flat or rising — memorization.)

✅ Next: [Module 9 — Fine-tuning + LoRA](../09_finetune_lora/README.md)
