import torch
from evaluate import evaluate_model
from inference import generate_text
import torch.optim as optim
from torch.optim import lr_scheduler
from train import train_epoch
from model import OLMo2Model
from torch.utils.data import DataLoader
from dataset import TextDataset
from transformers import AutoTokenizer
import torch.nn as nn
from tokenizer import DummyTokenizer

model_config = {
    "model_name": "OLMo2-Advanced-Version",
    "num_layers": 32,  # Original value
    "hidden_dim": 4096,  # Original value
    "num_attention_heads": 32,  # Original value
    "vocab_size": 50257,
    "context_window_size": 2048, # Original value
    "activation_function": "swiglu",
    "dropout_rate": 0.1,
    "positional_encoding": "rope",
    "normalization_layer": "rmsnorm",
    "attention_mechanism": {
        "type": "grouped_query_attention",
        "num_kv_heads": 8 # Original value
    },
    "mlp_ratio": 8/3,
    "use_bias_in_linear": False
}

# Reduce key parameters to prevent RAM deficiency as per instructions
model_config["hidden_dim"] = 512
model_config["num_layers"] = 4
model_config["num_attention_heads"] = 8
model_config["attention_mechanism"]["num_kv_heads"] = 1
model_config["context_window_size"] = 256

# Prepare sample data
sample_text = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly transforming various industries.",
    "Large language models like OLMo 2 are pushing the boundaries of natural language understanding and generation."
]

# Load a tokenizer that matches the vocab_size in model_config (e.g., GPT-2)
# The vocab_size for GPT-2 is 50257, which matches our model_config.

try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Use EOS token as pad token for GPT-2
    print(f"Tokenizer '{tokenizer.name_or_path}' loaded successfully.")
    print(f"Tokenizer vocab size: {len(tokenizer)} (expected {model_config['vocab_size']})")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure 'transformers' library is installed: pip install transformers")
    # Fallback to a dummy tokenizer or handle error gracefully
    tokenizer = DummyTokenizer(model_config['vocab_size'], model_config['context_window_size'])
    print("Using a dummy tokenizer due to error or missing installation.")

# 1. Use the tokenizer to encode the sample_text
encoded_inputs = tokenizer(
    sample_text,
    padding='max_length',
    truncation=True,
    max_length=model_config['context_window_size'],
    return_tensors='pt'
)

# 2. Extract input_ids and attention_mask
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

# 3. Create the labels tensor by making a copy of the input_ids tensor
labels = input_ids.clone()

# 4. Shift the labels tensor to the left by one position
# For each sequence, labels[i] should be input_ids[i+1]
# This means we take input_ids from the second token onwards, and append a pad_token_id at the end
labels = torch.cat([input_ids[:, 1:], torch.full((input_ids.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long)], dim=1)

dataset = TextDataset(input_ids, attention_mask, labels)

# Define batch_size. Using a small batch size for demonstration due to sample data size.
batch_size = 2 # Adjusted for demonstration and to prevent RAM issues with small dataset

# Instantiate the DataLoader
# Set num_workers=0 for simplicity in Colab, as multi-processing can sometimes be tricky.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
olmo2_model = OLMo2Model(model_config)

# Ensure model is on the correct device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
olmo2_model.to(device)

# 1. Define the criterion (loss function)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 2. Define the optimizer
# Using a small learning rate for demonstration, will be adjusted in training
lr = 1e-4
optimizer = optim.AdamW(olmo2_model.parameters(), lr=lr)


# Define the number of training epochs
num_epochs = 3 # Reduced for demonstration purposes

# Initialize a learning rate scheduler
# CosineAnnealingLR adjusts the learning rate using a cosine schedule
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Main Training Loop
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Train for one epoch
    avg_train_loss = train_epoch(olmo2_model, dataloader, criterion, optimizer, device)

    # Update the learning rate scheduler
    scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

print("Training complete.")

# 1. Call the evaluate_model function
val_loss, val_perplexity = evaluate_model(olmo2_model, dataloader, criterion, device)

# 2. Print the average loss and perplexity
print(f"\n--- Evaluation Results ---")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Perplexity: {val_perplexity:.2f}")


prompt_text = "Artificial intelligence is"
generated_output = generate_text(
    olmo2_model,
    tokenizer,
    prompt_text,
    max_new_tokens=50,
    temperature=0.7,
    device=device
)

print("\n--- Generated Text ---")
print("Prompt:", prompt_text)
print("Generated text:", generated_output)