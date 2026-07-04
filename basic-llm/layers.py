import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_dim = self.num_attention_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_dim)
        self.key = nn.Linear(config.hidden_size, self.all_head_dim)
        self.value = nn.Linear(config.hidden_size, self.all_head_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_dim**0.5)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in EncoderModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to encourage attention to not be too dependent on a few tokens.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.output(context_layer)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU() # Using GELU activation as it's common in modern transformers
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.feed_forward = FeedForwardNetwork(config)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention block
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm_1(hidden_states + attention_output) # Add & Norm

        # Feed-Forward block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm_2(hidden_states + feed_forward_output) # Add & Norm

        return hidden_states

