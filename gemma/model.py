from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class GemmaLikeModel(nn.Module):
  def __init__(self, 
               vocab_size:int,
               max_seq_len:int,
               embed_dim:int,
               num_heads:int,
               hidden_dim:int,
               num_layers:int,
               dropout_rate:float=0.1
               ):
    super().__init__()
    self.vocab_size=vocab_size
    self.max_seq_len=max_seq_len
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    self.hidden_dim=hidden_dim
    self.num_layers=num_layers
    self.dropout_rate=dropout_rate
    self.token_embeddings=nn.Embedding(vocab_size,embed_dim)
    self.positional_embeddings=nn.Embedding(max_seq_len,embed_dim)
    self.transformer_blocks=nn.ModuleList([
        TransformerBlock(embed_dim,num_heads,hidden_dim,dropout_rate)
        for _ in range(num_layers)
    ])
    self.final_norm=RMSNorm(embed_dim)
    self.output_projection=nn.Linear(embed_dim,vocab_size)

  def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor=None):
    seq_len=input_ids.size(1)
    if seq_len>self.max_seq_len:
      raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

    token_embeds=self.token_embeddings(input_ids)
    position_ids=torch.arange(0,seq_len,dtype=torch.long,device=input_ids.device)
    # Fix: Pass position_ids to positional_embeddings, not position_embeds
    position_embeds=self.positional_embeddings(position_ids)

    x=token_embeds+position_embeds

    for block in self.transformer_blocks:
      x=block(x,mask=attention_mask)

    return self.output_projection(self.final_norm(x))