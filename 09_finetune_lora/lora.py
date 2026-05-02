"""Module 9 — LoRA wrapper for nn.Linear."""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear and adds a low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)         # freeze original W (and bias)
        in_f, out_f = base.in_features, base.out_features
        self.r = r; self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))     # zero init for B
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # base output + low-rank delta
        out = self.base(x)
        delta = self.drop(x) @ self.A.T @ self.B.T       # (..., out_f)
        return out + self.scaling * delta


def replace_linears_with_lora(module: nn.Module, r=8, alpha=16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            replace_linears_with_lora(child, r, alpha)


def lora_trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "07_gpt_pytorch"))
    from gpt_solution import GPT, GPTConfig

    model = GPT(GPTConfig())
    n_total = sum(p.numel() for p in model.parameters())
    for p in model.parameters(): p.requires_grad_(False)
    replace_linears_with_lora(model, r=4)
    n_train = sum(p.numel() for p in lora_trainable_params(model))
    print(f"trainable: {n_train:,} / total: {n_total:,}  ({100*n_train/n_total:.2f}%)")
