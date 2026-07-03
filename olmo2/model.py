import torch
import torch.nn as nn
from layers import TransformerBlock,RMSNorm

class OLMo2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config # Store config for easy access to parameters
        self.vocab_size = config['vocab_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.context_window_size = config['context_window_size']
        self.num_attention_heads = config['num_attention_heads']

        # 1. Embedding layer
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)

        # 2. Stack of TransformerBlocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config['hidden_dim'],
                num_attention_heads=config['num_attention_heads'],
                num_kv_heads=config['attention_mechanism']['num_kv_heads'],
                mlp_ratio=config['mlp_ratio'],
                dropout_rate=config['dropout_rate'],
                context_window_size=config['context_window_size'],
                use_bias=config['use_bias_in_linear']
            )
            for _ in range(self.num_layers)
        ])

        # 3. Final RMSNorm layer
        self.final_norm = RMSNorm(self.hidden_dim)

        # 4. Language model head
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=config['use_bias_in_linear'])

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # 1. Convert input_ids into token embeddings
        x = self.token_embeddings(input_ids) # (batch_size, seq_len, hidden_dim)

        # 2. Generate a causal attention mask
        # Create a base causal mask (seq_len, seq_len) with -inf in upper triangle
        causal_mask_base = torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype)
        causal_mask_base = torch.triu(causal_mask_base, diagonal=1) # (seq_len, seq_len)

        # Expand the causal mask to (batch_size, num_attention_heads, seq_len, seq_len)
        # This ensures dim 1 matches attn_scores, avoiding potential broadcasting issues
        causal_mask = causal_mask_base.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_attention_heads, 1, 1)

        # If an external attention_mask is provided (e.g., for padding), combine it with the causal mask
        if attention_mask is not None:
            # attention_mask from dataloader is (batch_size, seq_len)
            # Create an additive padding mask: (batch_size, 1, 1, seq_len)
            padding_mask_additive = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(x.dtype).min

            # Add padding mask to the pre-expanded causal mask.
            # Causal_mask is (B, H, S, S), padding_mask_additive is (B, 1, 1, S)
            # This addition will broadcast padding_mask_additive across heads and query positions.
            causal_mask = causal_mask + padding_mask_additive

        # 3. Pass through the stack of TransformerBlocks
        for layer in self.layers:
            x = layer(x, attention_mask=causal_mask)

        # 4. Apply the final RMSNorm layer
        x = self.final_norm(x)

        # 5. Pass through the language model head to get the final logits
        logits = self.lm_head(x)

        return logits