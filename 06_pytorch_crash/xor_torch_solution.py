"""Solution."""

import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(1337)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


model = MLP()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(2000):
    pred = model(X); loss = F.mse_loss(pred, Y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.6f}")

with torch.no_grad():
    print(model(X).squeeze().tolist())
