import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # RMSNorm is computed as: x * rsqrt(mean(x^2) + eps)
        # where x is hidden_states, and mean is across the last dimension (hidden_size)
        rms = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return hidden_states * rms * self.weight

def SwiGLU(x):
    split_dim = x.shape[-1] // 2
    x1, x2 = x[..., :split_dim], x[..., split_dim:]
    swish_x1 = F.silu(x1)
    return swish_x1 * x2

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio, use_bias, dropout_rate):
        super().__init__()
        # The first linear layer's output dimension, which is then split by SwiGLU.
        # This ensures that the effective intermediate dimension after SwiGLU
        # is (hidden_dim * mlp_ratio / 2).
        # We ensure ffn_hidden_dim is even for SwiGLU split.
        self.ffn_hidden_dim = int(hidden_dim * mlp_ratio)
        if self.ffn_hidden_dim % 2 != 0:
            self.ffn_hidden_dim = (self.ffn_hidden_dim // 2) * 2 # Make it an even number

        self.w1 = nn.Linear(hidden_dim, self.ffn_hidden_dim, bias=use_bias)
        # The output of SwiGLU will be self.ffn_hidden_dim // 2
        self.w2 = nn.Linear(self.ffn_hidden_dim // 2, hidden_dim, bias=use_bias)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        x = self.w1(x) # (batch_size, seq_len, ffn_hidden_dim)
        x = SwiGLU(x) # (batch_size, seq_len, ffn_hidden_dim // 2)
        x = self.dropout(x)
        x = self.w2(x) # (batch_size, seq_len, hidden_dim)
        return x
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_kv_heads, dropout_rate, context_window_size, use_bias):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_attention_heads
        self.scale = self.head_dim ** -0.5

        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_groups = self.num_attention_heads // self.num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_attention_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_dim, bias=use_bias)

        self.rope = RotaryPositionalEmbedding(self.head_dim, context_window_size)
        self.attn_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.res_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        # q_proj_output: (batch_size, seq_len, num_attention_heads * head_dim)
        # kv_proj_output: (batch_size, seq_len, num_kv_heads * head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch_size, num_heads/num_kv_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        # q_rotated, k_rotated: (batch_size, num_heads/num_kv_heads, seq_len, head_dim)
        q_rotated, k_rotated = self.rope(q, k, seq_len)

        # Grouped Query Attention: Replicate K and V heads if num_kv_heads < num_attention_heads
        # K and V need to be replicated to match the number of query heads
        if self.num_groups > 1:
            k_rotated = k_rotated.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Compute attention scores
        # (batch_size, num_attention_heads, seq_len, head_dim) @ (batch_size, num_attention_heads, head_dim, seq_len)
        # -> (batch_size, num_attention_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) * self.scale

        # Apply attention mask (causal mask for decoder-only transformers)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention weights to values
        # (batch_size, num_attention_heads, seq_len, seq_len) @ (batch_size, num_attention_heads, seq_len, head_dim)
        # -> (batch_size, num_attention_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project back to hidden_dim
        # (batch_size, seq_len, num_attention_heads, head_dim) -> (batch_size, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attn_output)
        output = self.res_dropout(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_kv_heads, mlp_ratio, dropout_rate, context_window_size, use_bias):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads

        # Pre-normalization for attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            context_window_size=context_window_size,
            use_bias=use_bias
        )

        # Pre-normalization for FFN
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len, hidden_dim)

        # Attention sub-layer with pre-normalization and residual connection
        h = self.attn_norm(x)
        h = self.attn(h, attention_mask=attention_mask)
        x = x + h # Residual connection

        # FFN sub-layer with pre-normalization and residual connection
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h # Residual connection

        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        # Initialize theta values for rotary embeddings
        # inv_freq should be of dim // 2, as it represents frequencies for pairs of dimensions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs will be (max_seq_len, dim // 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb will be (max_seq_len, dim) where the frequencies are duplicated for each half
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def _rotate_half(self, x):
        # Splits x into two halves along the last dimension and rotates them
        # e.g., for x = [a, b, c, d], returns [-c, -d, a, b]
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward_applying_rope(self, x, seq_len=None):
        # x: (batch_size, num_heads, seq_len, head_dim)
        # Retrieve precomputed cos and sin values for the current sequence length
        cos = self.cos_cached[:, :, :seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.device)

        # Apply rotary embeddings: x_rotated = x * cos + rotate_half(x) * sin
        # All tensors (x, cos, sin, _rotate_half(x)) will have the same last dimension (head_dim),
        # resolving the dimension mismatch error.
        x_rotated_half = self._rotate_half(x)
        return x * cos + x_rotated_half * sin

    def forward(self, q, k, seq_len):
        # q, k: (batch_size, num_heads, seq_len, head_dim)
        q_rotated = self.forward_applying_rope(q, seq_len)
        k_rotated = self.forward_applying_rope(k, seq_len)
        return q_rotated, k_rotated