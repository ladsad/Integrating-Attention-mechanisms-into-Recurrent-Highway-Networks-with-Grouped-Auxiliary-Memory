import torch
import torch.nn as nn
from rhn_cell import RHNCell
from config import DEVICE

class GAM_RHN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(GAM_RHN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding Layer
        self.rhn_cell = RHNCell(embedding_dim, hidden_size)  # RHN Cell
        self.fc = nn.Linear(hidden_size, num_classes)  # Output layer (from hidden_size to num_classes)

    def forward(self, x):

        # Get the embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Initialize hidden state
        h = torch.zeros(x.size(0), self.rhn_cell.hidden_size).to(DEVICE)  # Shape: (batch_size, hidden_size)

        # Iterate over the sequence
        for step in range(x.size(1)):  # x.size(1) is the sequence length
            h = self.rhn_cell(embedded[:, step, :], h)  # Shape: (batch_size, hidden_size)

        # Return final output after passing through the fully connected layer
        output = self.fc(h)  # Shape: (batch_size, num_classes)

        return output