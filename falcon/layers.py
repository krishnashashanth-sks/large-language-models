import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TokenEmbeddings (re-defining for completeness, no changes)
class TokenEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])

    def forward(self, input_ids):
        return self.token_embeddings(input_ids)

# RotaryPositionalEmbedding (corrected)
class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, head_dim, config):
    super().__init__()
    self.dim = head_dim
    self.seq_len_interpolation_factor = 1 # Keep at 1 for now, can be changed for RoPE scaling
    self.rope_theta = config.get("rope_theta", 10000.0) # Corrected typo from 'rope_theat' and default
    inv_freq = 1.0 / (self.rope_theta**(torch.arange(0, self.dim, 2).float() / self.dim))
    self.register_buffer("inv_freq", inv_freq)
    self.max_seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def _update_cos_sin_cache(self, x, seq_len=None):
    if seq_len is None:
      # If seq_len not provided, infer from the input tensor x
      # Assuming x is (batch_size, num_heads, seq_len, head_dim)
      seq_len = x.shape[-2]

    # Corrected condition to handle None and update cache if new seq_len is greater
    if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
      self.max_seq_len_cached = seq_len
      # Re-calculate t to match the new sequence length
      t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
      t = t / self.seq_len_interpolation_factor
      freqs = torch.outer(t, self.inv_freq)
      emb = torch.cat((freqs, freqs), dim=-1) # Concat for real and imaginary parts
      # Store as (1, 1, seq_len, dim) for broadcasting
      self.cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
      self.sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

  def rotate_half(self, x):
    x1 = x[..., :self.dim // 2]
    x2 = x[..., self.dim // 2:]
    return torch.cat((-x2, x1), dim=-1)

  def apply_rotary_pos_emb(self, q, k, positions=None):
    # q, k shape: (batch_size, num_heads, seq_len, head_dim)
    current_seq_len = q.shape[-2]
    self._update_cos_sin_cache(q, seq_len=current_seq_len) # Update cache based on current sequence length

    # Apply RoPE using cached sin and cos values, slicing to current_seq_len
    q_embed = (q * self.cos_cached[:, :, :current_seq_len, :]) + (self.rotate_half(q) * self.sin_cached[:, :, :current_seq_len, :])
    k_embed = (k * self.cos_cached[:, :, :current_seq_len, :]) + (self.rotate_half(k) * self.sin_cached[:, :, :current_seq_len, :])
    return q_embed, k_embed

# MultiQueryAttention (re-defining for completeness, minor adjustment for error message)
class MultiQueryAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.hidden_size = config['hidden_size']
    self.num_attention_heads = config["num_attention_heads"]
    self.num_kv_heads = config["num_kv_heads"]
    self.head_dim = self.hidden_size // self.num_attention_heads
    self.bias = config['bias']

    if (self.head_dim * self.num_attention_heads) != self.hidden_size:
      raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_attention_heads})."
      )
    self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias)
    self.kv_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim * 2, bias=self.bias)
    self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias)
    self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config)

  def _split_heads(self, hidden_states, num_heads):
    new_shape = hidden_states.shape[:-1] + (num_heads, self.head_dim)
    return hidden_states.view(new_shape).permute(0, 2, 1, 3)

  def _merge_heads(self, hidden_states):
    hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
    batch_size, seq_len, _, _ = hidden_states.shape
    return hidden_states.view(batch_size, seq_len, self.num_attention_heads * self.head_dim)

  def forward(self, hidden_states, attention_mask=None):
    batch_size, seq_len, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    kv_states = self.kv_proj(hidden_states)
    query_states = self._split_heads(query_states, self.num_attention_heads)

    kv_states = kv_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
    key_states = kv_states[:, :, :, :self.head_dim]
    value_states = kv_states[:, :, :, self.head_dim:]

    key_states = key_states.permute(0, 2, 1, 3)
    value_states = value_states.permute(0, 2, 1, 3)

    if self.rotary_emb is not None:
      query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(query_states, key_states)

    if self.num_kv_heads < self.num_attention_heads:
      key_states = key_states.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=1)
      value_states = value_states.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=1)

    attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
      attn_scores = attn_scores + attention_mask
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = self._merge_heads(attn_output)
    output = self.o_proj(attn_output)
    return output

# FFN (re-defining for completeness, no changes)
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.hidden_size = config['hidden_size']
    self.intermediate_size = config['intermediate_size']
    self.bias = config['bias']
    self.drop_rate = config['dropout_rate']
    self.dense_in = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.bias)
    self.gelu = nn.GELU()
    self.dense_out = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.bias)
    self.dropout = nn.Dropout(self.drop_rate)
  def forward(self, hidden_states):
    return self.dropout(self.dense_out(self.gelu(self.dense_in(hidden_states))))

# LayerNorm (re-defining for completeness, no changes)
class LayerNorm(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layer_norm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_epsilon"])
  def forward(self, hidden_states):
    return self.layer_norm(hidden_states)

# FalconDecoderBlock (corrected for parallel attention residual connection)
class FalconDecoderBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.input_layernorm = LayerNorm(config) # Pre-attention and Pre-FFN LayerNorm
    self.self_attention = MultiQueryAttention(config)
    self.mlp = FFN(config)
    self.dropout = nn.Dropout(config['dropout_rate'])

  def forward(self, hidden_states, attention_mask=None):
    # Falcon's parallel attention applies LN once, then forks to attention and MLP
    layernorm_output = self.input_layernorm(hidden_states)

    # Parallel branches
    attn_output = self.self_attention(layernorm_output, attention_mask=attention_mask)
    mlp_output = self.mlp(layernorm_output) # MLP also takes layernorm_output

    # Sum of attention and MLP outputs, then residual connection with dropout
    output = hidden_states + self.dropout(attn_output + mlp_output)
    return output