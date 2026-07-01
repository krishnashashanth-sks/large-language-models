import torch
from torch.utils.data import DataLoader
from dataset import LanguageModelingDataset
from tokenizer import CharTokenizer
from train import train
from inference import generate_text
import os
from utils import create_sequences
import torch.nn as nn
from model import Falcon

model_config = {
    "vocab_size": 65024,  # Common for Falcon models, adjust based on specific tokenizer
    "num_hidden_layers": 32, # Number of transformer layers
    "hidden_size": 4544, # Dimensionality of the embedding space and hidden states
    "num_attention_heads": 32, # Number of attention heads
    "num_kv_heads": 32, # Number of key/value heads for multi-query or grouped-query attention
    "intermediate_size": 12288, # Size of the intermediate feed-forward network
    "multi_query_attention": False, # Whether to use multi-query attention (True for many Falcon versions)
    "parallel_attention": True, # Whether to use parallel attention layers (True for Falcon)
    "alibi": False, # Whether to use ALiBi (Attention with Linear Biases) for positional embeddings
    "rope_theta": 10000.0, # Base value for RoPE (Rotary Positional Embeddings)
    "bias": False, # Whether to use bias in linear layers
    "layer_norm_epsilon": 1e-05, # Epsilon for layer normalization
    "initializer_range": 0.02, # Standard deviation for weight initialization
    "dropout_rate": 0.1, # Dropout probability
    "max_position_embeddings": 2048 # Maximum sequence length the model can handle
}

model_config.update({
    "num_hidden_layers": 4,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_kv_heads": 1,
    "intermediate_size": 512,
    "max_position_embeddings": 128
})

raw_text_data = """The Falcon large language model is a powerful transformer architecture developed by Technology Innovation Institute (TII). It leverages unique features like Multi-Query Attention and parallel attention layers for efficiency and performance. This model is designed for various natural language processing tasks, including text generation, summarization, and question answering. Training such a model from scratch requires careful preparation of large datasets and efficient hardware. The architecture's design emphasizes scalability and optimization for distributed training environments. Multi-Query Attention helps reduce memory footprint and improve inference speed, making Falcon an attractive choice for deploying large-scale AI applications."""

# Initialize the character tokenizer with the raw text data
char_tokenizer = CharTokenizer(raw_text_data)

# Update vocab_size in model_config to match the character tokenizer's vocab_size
model_config["vocab_size"] = char_tokenizer.vocab_size

block_size = 128 # A smaller, more suitable block size for the given text length

encoded_text = char_tokenizer.encode(raw_text_data)

input_data, target_data = create_sequences(encoded_text, block_size)

dataset = LanguageModelingDataset(input_data, target_data)


# Define DataLoader parameters
batch_size = 8  # Choose a suitable batch size
shuffle = True  # Shuffle data for training
num_workers = os.cpu_count() # Number of subprocesses to use for data loading (0 means main process)

# Create DataLoader instances
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Falcon(model_config).to(device)
learning_rate = 1e-4 # Ensure this is consistent with previous definition
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()
print("CrossEntropyLoss function initialized.")

num_epochs = 5  # Number of training epochs
gradient_clipping_value = 1.0 # Value for gradient clipping

train(num_epochs,train_dataloader,model,optimizer,loss_fn,gradient_clipping_value,device)

print("\n--- Starting Evaluation ---")

model.eval()  # Set the model to evaluation mode
total_eval_loss = 0

with torch.no_grad():  # Disable gradient calculations during evaluation
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)

        # Calculate loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_eval_loss += loss.item()

# Calculate average loss and perplexity
avg_eval_loss = total_eval_loss / len(train_dataloader)
perplexity = torch.exp(torch.tensor(avg_eval_loss))

prompt_text = "The Falcon model is a powerful"
generated_output = generate_text(model, char_tokenizer, prompt_text, max_new_tokens=50, temperature=0.7, device=device)
print(f"Prompt: {prompt_text}")
print(f"Generated: {generated_output}")