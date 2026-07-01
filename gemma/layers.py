import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        # 3a, 3b, 3c: Initialize parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # 3d: Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.head_dim = embed_dim // num_heads

        # 4: Define linear projection layers for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 5: Define a linear projection layer for the output
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 6b: Project the input x into query, key, and value tensors
        q = self.q_proj(x) # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x) # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x) # (batch_size, seq_len, embed_dim)

        # 6c: Reshape the projected query, key, and value tensors
        #    to separate them into multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q, k, v shape: (batch_size, num_heads, seq_len, head_dim)

        # 6d: Implement the scaled dot-product attention mechanism
        # Calculate dot products (Q @ K.T)
        # attn_scores shape: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to be broadcastable to (batch_size, num_heads, seq_len, seq_len)
            # The original mask is typically (batch_size, seq_len)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2) # Shape becomes (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(expanded_mask == 0, float('-inf'))

        # Apply softmax function to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 6e: Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # 6f: Compute the weighted sum of the value tensors (attn_weights @ V)
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        # 6g: Concatenate the outputs from all heads
        # Transpose back to (batch_size, seq_len, num_heads, head_dim) then flatten the last two dims
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 6h: Apply the final output projection layer
        output = self.out_proj(output)

        return output


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()

        # 2a: Input linear layer
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        # 2b: Gate linear layer
        self.w2 = nn.Linear(embed_dim, hidden_dim)
        # 2d: Output linear layer
        self.w3 = nn.Linear(hidden_dim, embed_dim)

        # 2e: Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 3a: Apply input linear layer to x
        hidden_input = self.w1(x)
        # 3b: Apply gate linear layer to x
        gate = self.w2(x)

        # 3c: Multiply the output of the input linear layer with the Swish activation
        #     applied to the output of the gate linear layer. (SwiGLU)
        gated_activation = hidden_input * F.silu(gate)

        # 3d: Apply dropout to the result
        gated_activation = self.dropout(gated_activation)

        # 3e: Pass the output through the final output linear layer
        output = self.w3(gated_activation)

        return output


# RMSNorm implementation as specified
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # The weight parameter for scaling. Initialized to ones.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Compute RMS (Root Mean Square) for the last dimension
        # x.pow(2).mean(-1, keepdim=True) calculates mean of squares
        # torch.rsqrt computes 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Apply RMS normalization and then scale by learned weight
        # .float() and .type_as(x) are for mixed precision training compatibility
        return self._norm(x.float()).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super().__init__()

        # 2a: Instance of MultiHeadSelfAttention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)

        # 2b: Instance of SwiGLUFFN
        self.feed_forward = SwiGLUFFN(embed_dim, hidden_dim, dropout_rate)

        # 2c: Two RMSNorm layers
        self.norm1 = RMSNorm(embed_dim) # For attention sub-layer
        self.norm2 = RMSNorm(embed_dim) # For FFN sub-layer

        # 2d: Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # 4a: Apply the first RMSNorm layer to the input x
        norm_x = self.norm1(x)

        # 4b: Pass the normalized input through the MultiHeadSelfAttention layer, adding a residual connection.
        attn_output = self.attention(norm_x, mask=mask)
        # 4c: Apply dropout to the output of the attention layer.
        attn_output = self.dropout(attn_output)
        # Residual connection
        x = x + attn_output

        # 4d: Apply the second RMSNorm layer
        norm_x = self.norm2(x)

        # 4e: Pass the normalized input through the SwiGLUFFN layer, adding another residual connection.
        ffn_output = self.feed_forward(norm_x)
        # 4f: Apply dropout to the output of the FFN layer.
        ffn_output = self.dropout(ffn_output)
        # Residual connection
        x = x + ffn_output

        # 4g: Return the final output of the transformer block.
        return x
