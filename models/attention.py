import torch
import torch.nn as nn

class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(DotProductAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, query, key, value):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value), attention_weights
    
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, query, key, value):
        # Repeat query to match the sequence length of key
        query = query.repeat(1, key.size(1), 1)  # Shape: (batch_size, sequence_length, hidden_size)

        # Concatenate query and key along the last dimension
        attn_input = torch.cat((query, key), dim=-1)  # Shape: (batch_size, sequence_length, hidden_size * 2)

        # Linear transformation for attention weights
        attn_weights = torch.tanh(self.attn(attn_input))  # Shape: (batch_size, sequence_length, hidden_size)
        attn_weights = torch.matmul(attn_weights, self.v.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, sequence_length)

        # Softmax to normalize the weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # Shape: (batch_size, sequence_length)

        # Compute the weighted sum of values based on the attention weights
        context = torch.matmul(attn_weights.unsqueeze(1), value)  # Shape: (batch_size, 1, hidden_size)

        return context, attn_weights
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, query, key, value):
        # Scale the dot product of query and key by the square root of hidden_size
        scale = self.hidden_size ** 0.5
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / scale  # Shape: (batch_size, query_len, key_len)

        # Softmax to get attention weights
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # Shape: (batch_size, query_len, key_len)

        # Multiply the attention weights by the value
        output = torch.matmul(attention_weights, value)  # Shape: (batch_size, query_len, hidden_size)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear layers to project queries, keys, and values to multiple heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Final linear layer after concatenating all heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Project and reshape queries, keys, and values for multi-head attention
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention on each head
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        # Multiply the attention weights by the values
        multi_head_output = torch.matmul(attention_weights, value)  # Shape: (batch_size, num_heads, query_len, head_dim)

        # Concatenate all heads and project back to the original hidden size
        multi_head_output = multi_head_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Apply final projection
        output = self.out_proj(multi_head_output)

        return output, attention_weights
