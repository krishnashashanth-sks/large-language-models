from main import tokenizer

class Config:
    vocab_size = tokenizer.vocab_size # Using the vocab size from our trained tokenizer
    max_position_embeddings = 512 # Maximum sequence length the model can handle
    hidden_size = 768 # Dimensionality of the embeddings and transformer layers
    num_layers = 6 # Number of transformer layers
    num_attention_heads = 12 # Number of attention heads in each transformer layer
    intermediate_size = hidden_size * 4 # Dimensionality of the "intermediate" (i.e., feed-forward) layer
    dropout = 0.1 # Dropout probability
    pad_token_id = tokenizer.pad_token_id # Padding token ID

    def __repr__(self):
        return f"Config(\n" \
               f"  vocab_size={self.vocab_size},\n" \
               f"  max_position_embeddings={self.max_position_embeddings},\n" \
               f"  hidden_size={self.hidden_size},\n" \
               f"  num_layers={self.num_layers},\n" \
               f"  num_attention_heads={self.num_attention_heads},\n" \
               f"  intermediate_size={self.intermediate_size},\n" \
               f"  dropout={self.dropout},\n" \
               f"  pad_token_id={self.pad_token_id}\n)"

