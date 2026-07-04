import torch.nn as nn
import math
from layers import PositionalEncoding

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_sequence_length):
        super(TransformerDecoderModel, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt_mask=None):
        embedded_src = self.token_embedding(src) * math.sqrt(self.d_model)
        pos_encoded_src = self.positional_encoding(embedded_src)
        decoder_output = self.transformer_decoder(pos_encoded_src, memory=pos_encoded_src, tgt_mask=tgt_mask)
        normalized_output = self.norm(decoder_output)
        output_logits = self.linear(normalized_output)
        return output_logits