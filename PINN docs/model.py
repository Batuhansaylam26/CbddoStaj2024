import torch
import torch.nn as nn
import torch.nn.init as init


class PinnNet(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        num_layers: int = 1,
        hidden_size: int = 5,
        output_size: int = 1,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        # layers list for sequantial
        layers = list()
        # input layer
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        # Hidden layers w/linear layer & activation
        for i in range(num_layers):
            layers.extend([nn.Linear(self.hidden_size, self.hidden_size), activation])
        # output layer
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.network(x)
