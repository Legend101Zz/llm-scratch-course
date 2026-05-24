"""Phase 0 capstone — train the TorchGPT on tinyshakespeare.

This is the *training* step. Module 1's Value engine could in principle train
this model, but each forward pass on a (T=64, d=64, 2-layer) GPT would touch
hundreds of thousands of scalar Value ops per step — minutes of wall time per
step. So we use PyTorch's vector autograd (which Modules 1, 4, 5's parity tests
proved is doing the same math as ours, just batched on arrays).

Train recipe (chosen to converge quickly on CPU):
  - char-level vocab (65 unique chars from tinyshakespeare)
  - max_T = 64    (context window)
  - d = 64, h = 4, n_layers = 2
  - batch size = 32
  - lr = 3e-3, AdamW, no warmup (model is too small to need it)
  - ~2000 steps on CPU = ~3–5 min on a modern laptop

Convergence target: train loss should drop from ~ln(65)≈4.17 (random) to under
2.0 nats within 1000 steps. Generations should start to look "Shakespearean"
in the sense that you see capital letters, line breaks, and short words.

Run:
    python train.py

Outputs (committed to ./evidence/):
  - evidence/loss_curve.png
  - evidence/sample_generations.txt
  - evidence/training_log.txt
"""

import os
import time
import json
import argparse
import math
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from torch_gpt import TorchGPT, count_parameters


HERE = os.path.dirname(__file__)
EVIDENCE_DIR = os.path.join(HERE, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)


def load_data():
    path = os.path.join(HERE, "..", "03_tokenizer_bigram", "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    text = open(path).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, stoi, itos, len(chars)


def get_batch(data: np.ndarray, batch_size: int, T: int, device: torch.device):
    """Random crops of length T+1 → inputs (T) + targets (T)."""
    starts = np.random.randint(0, len(data) - T - 1, size=batch_size)
    xs = np.stack([data[s : s + T] for s in starts])
    ys = np.stack([data[s + 1 : s + 1 + T] for s in starts])
    return torch.from_numpy(xs).long().to(device), torch.from_numpy(ys).long().to(device)


@torch.no_grad()
def estimate_loss(model: TorchGPT, data: np.ndarray, batch_size: int, T: int,
                  device: torch.device, n_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, T, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data, stoi, itos, V = load_data()
    print(f"data: {len(data):,} tokens  vocab: {V}")

    # Split 90/10 for sanity (won't actually use val_data heavily, just for the
    # estimate_loss eval-set call).
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    device = torch.device(args.device)
    model = TorchGPT(V, args.d, args.h, args.n_layers, args.T).to(device)
    print(f"model params: {count_parameters(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    log = {"step": [], "train_loss": [], "val_loss": []}
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_data, args.bs, args.T, device)
        _, loss = model(x, targets=y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss = estimate_loss(model, val_data, args.bs, args.T, device)
            elapsed = time.time() - t0
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["val_loss"].append(val_loss)
            print(f"step {step:5d}/{args.steps}  train {loss.item():.3f}  val {val_loss:.3f}  "
                  f"({elapsed:.1f}s elapsed)")

    # ----- artifacts -----
    # Loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(log["step"], log["train_loss"], label="train")
        plt.plot(log["step"], log["val_loss"], label="val")
        plt.axhline(math.log(V), color="grey", linestyle=":", label=f"uniform = ln({V})")
        plt.xlabel("step"); plt.ylabel("loss (nats)"); plt.legend()
        plt.title(f"Phase 0 capstone — tiny GPT on tinyshakespeare ({args.steps} steps)")
        plt.savefig(os.path.join(EVIDENCE_DIR, "loss_curve.png"), dpi=100, bbox_inches="tight")
        print(f"saved evidence/loss_curve.png")
    except ImportError:
        print("(matplotlib not available — skipping loss curve plot)")

    # Training log
    with open(os.path.join(EVIDENCE_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("saved evidence/training_log.json")

    # Generations
    model.eval()
    samples_lines = []
    prompts = ["ROMEO:", "\n", "JULIET:"]
    with torch.no_grad():
        for prompt in prompts:
            if not prompt:
                continue
            try:
                ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
            except KeyError:
                continue
            out = model.generate(ids, max_new_tokens=200, temperature=0.8)
            decoded = "".join(itos[i] for i in out[0].tolist())
            samples_lines.append(f"=== prompt: {prompt!r} ===")
            samples_lines.append(decoded)
            samples_lines.append("")
    with open(os.path.join(EVIDENCE_DIR, "sample_generations.txt"), "w") as f:
        f.write("\n".join(samples_lines))
    print("saved evidence/sample_generations.txt")

    # Final summary
    print()
    print(f"final train loss: {log['train_loss'][-1]:.3f}")
    print(f"final val   loss: {log['val_loss'][-1]:.3f}")
    print(f"uniform baseline: {math.log(V):.3f}  (= ln(vocab_size))")
    print(f"wall time: {time.time() - t0:.1f}s on {device}")


if __name__ == "__main__":
    main()
