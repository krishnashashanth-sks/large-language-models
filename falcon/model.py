from layers import *

# Falcon (re-defining for completeness, improved attention mask generation)
class Falcon(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.token_embeddings = TokenEmbeddings(config)
    self.h = nn.ModuleList([
        FalconDecoderBlock(config) for _ in range(config["num_hidden_layers"])
    ])
    self.ln_f = LayerNorm(config)
    self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

  def forward(self, input_ids, attention_mask=None):
    hidden_states = self.token_embeddings(input_ids)

    # Create a causal mask if no specific attention_mask is passed, or if it's implicitly True
    if attention_mask is not None:
      seq_len = input_ids.shape[1]
      # Create a causal mask (look-ahead mask) with large negative values for masked positions
      attention_mask = torch.triu(
          torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=hidden_states.device),
          diagonal=1
      ).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, seq_len, seq_len) for broadcasting

    for layer_module in self.h:
      hidden_states = layer_module(hidden_states, attention_mask=attention_mask)

    return self.lm_head(self.ln_f(hidden_states))
