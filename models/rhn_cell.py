import torch
import torch.nn as nn

class RHNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RHNCell, self).__init__()
        self.hidden_size = hidden_size
        self.rhn_layers = 5  # Number of recurrent layers
        self.input_size = input_size

        # Projection layer to match the input size to the hidden size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Hidden layers for each recurrent step (they should accept hidden_size as input and output)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.rhn_layers)]
        )

    def forward(self, x, h):

        # Project input to match the hidden size (from input_size to hidden_size)
        x = self.input_projection(x)  # Shape: (batch_size, hidden_size)

        # Recurrent Highway Network with multiple layers
        for layer in self.hidden_layers:
            h = torch.tanh(layer(x) + h)  # Shape: (batch_size, hidden_size)

        return h