import torch.nn as nn
from layers import Embeddings, TransformerBlock
import torch

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        # Create a causal attention mask (look-ahead mask)
        if attention_mask is None:
            seq_len = input_ids.size(1)
            # Lower triangular matrix with 1s, others 0s
            # This creates a mask where position i can only attend to positions j <= i
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
            # Expand for batch size and number of heads
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            # Convert to float and replace 0s with -inf for additive mask
            attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
        else:
            # If an attention_mask is provided (e.g., for padding), combine it with the causal mask
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = combined_mask.float().masked_fill(combined_mask == 0, float('-inf')).masked_fill(combined_mask == 1, float(0.0))

        hidden_states = self.embeddings(input_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)

        logits = self.lm_head(hidden_states)

        return logits

