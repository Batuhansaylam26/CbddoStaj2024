from LossFunction import LossFunction
from model import PinnNet
from displacement import Displacement
import torch
import numpy as np

model = PinnNet()
omega = torch.from_numpy(np.linspace(0.001, 12, 25)).reshape(-1, 1).requires_grad_(True)
omegaBoundary = torch.tensor(0.0).view(-1, 1).requires_grad_(True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 15001  # number of training iterations per batch of data
losses = []
losses_physic = []
losses_ic = []
results = []
loss=LossFunction()
# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss(t_train, t_boundary, model)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    results.append(model(t_train).detach().numpy())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
