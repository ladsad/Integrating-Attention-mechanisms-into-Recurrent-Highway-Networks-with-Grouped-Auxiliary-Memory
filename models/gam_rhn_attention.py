import torch
import torch.nn as nn
from rhn_cell import RHNCell
from attention import DotProductAttention, AdditiveAttention, ScaledDotProductAttention, MultiHeadAttention
from config import DEVICE

class GAM_RHN_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, attention_type='dot_product', num_heads=8):
        super(GAM_RHN_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rhn_cell = RHNCell(embedding_dim, hidden_size)
        self.attention_type = attention_type

        # Initialize different attention mechanisms based on attention_type
        if attention_type == 'dot_product':
            self.attention = DotProductAttention(hidden_size)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(hidden_size)
        elif attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(hidden_size)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadAttention(hidden_size, num_heads)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        h = torch.zeros(x.size(0), self.rhn_cell.hidden_size).to(DEVICE)  # Initialize hidden state

        # Process the input sequence through RHN cell and collect hidden states
        rhn_outputs = []  # To store hidden states from each time step
        for step in range(x.size(1)):
            h = self.rhn_cell(embedded[:, step, :], h)  # Forward step through RHN cell
            rhn_outputs.append(h.unsqueeze(1))  # Shape: (batch_size, 1, hidden_size)

        rhn_outputs = torch.cat(rhn_outputs, dim=1)  # Shape: (batch_size, sequence_length, hidden_size)

        # Attention mechanism: Query is the last hidden state, Key and Value are the entire sequence
        query = h.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        key = rhn_outputs  # Shape: (batch_size, sequence_length, hidden_size)
        value = rhn_outputs  # Shape: (batch_size, sequence_length, hidden_size)

        # Apply the selected attention mechanism
        context, attn_weights = self.attention(query, key, value)
        context = context.squeeze(1)  # Remove the singleton dimension after attention

        # Output prediction via fully connected layer
        output = self.fc(context)  # Shape: (batch_size, num_classes)

        return output, attn_weights
