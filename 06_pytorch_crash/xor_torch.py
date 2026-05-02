"""Module 6 — XOR MLP in PyTorch. Compare to course/02_neural_net/solution.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: two hidden Linear(2->4), Linear(4->4), output Linear(4->1)
        # self.l1 = nn.Linear(2, 4); self.l2 = nn.Linear(4, 4); self.l3 = nn.Linear(4, 1)
        pass

    def forward(self, x):
        # TODO: tanh after l1, tanh after l2, raw output l3
        pass


model = MLP()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2000):
    pred = model(X)
    loss = F.mse_loss(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.6f}")

with torch.no_grad():
    print(model(X).squeeze().tolist())
