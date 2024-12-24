import torch as torch
import numpy as np
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(
        self,
        omegas: torch.tensor,
        omegas_boundary: torch.tensor,
        model: torch.nn.Module,
    ) -> None:
        self.omegas = omegas
        self.omegas_boundary = omegas_boundary
        self.model = model

    # compute predictions
    def xPred(self) -> torch.tensor:
        return self.model(self.omegas)

    def xboundary(self) -> torch.tensor:
        return self.model(self.omegas_boundary)

    # Compute gradients of each output wrt t,boundary t
    def gd2xdt2Pred(self) -> torch.tensor:
        xPred = self.xPred()
        dxdt = torch.autograd.grad(xPred, self.omegas, torch.ones_like(xPred))[0]
        return torch.autograd.grad(
            dxdt, self.omegas, torch.ones_like(dxdt), create_graph=True
        )[0]

    def resiudal(
        self, c_R: float, k1: torch.tensor, k2: torch.tensor, mu: float
    ) -> torch.tensor:
        self.B = k1 / k2 * (1 - k1**2) + k2 / k1 * (1 - k2**2) - (1 - k2**4)
        residual = (
            self.xPred()
            - (1 / c_R**2) * self.gd2xdt2Pred()
            + k1 * (1 - k2**2) / (4 * mu * self.B)
        )
        return residual

    def forward(self, c_R: float, k1: torch.tensor, k2: torch.tensor, mu: float):
        loss = nn.MSELoss()
        res = self.resiudal(c_R, k1, k2, mu)
        return loss(res, torch.zeros_like(res))
