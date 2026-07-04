from model import DecoderOnlyTransformer
from train import train_epoch
from evaluate import evaluate
from inference import generate
import torch
import torch.nn as nn
from config import Config
from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import os
from datasets import load_dataset
from utils import get_training_corpus,group_texts

print("Loading wikitext-103-raw-v1 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

print("Dataset loaded successfully:")
print(dataset)
print("First entry of the training split:")
print(dataset["train"][0])

# Initialize a ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the corpus using train_from_iterator
print("Training the tokenizer from iterator...")
tokenizer.train_from_iterator(
    get_training_corpus(dataset), # Pass the iterator as the first positional argument
    vocab_size=50257, # Standard vocab size for models like GPT-2
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)
print("Tokenizer training complete.")

# Save the tokenizer
tokenizer_dir = "my_tokenizer"
if not os.path.exists(tokenizer_dir):
    os.makedirs(tokenizer_dir)

tokenizer.save_model(tokenizer_dir)

# Explicitly save the tokenizer as tokenizer.json for PreTrainedTokenizerFast
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
print(f"Tokenizer saved to {tokenizer_dir}")
# Load the tokenizer
# We need to specify the tokenizer file (vocab.json and merges.txt) within the directory
# For ByteLevelBPETokenizer, it saves vocab.json and merges.txt

# Check if the tokenizer files exist
vocab_file = os.path.join(tokenizer_dir, "vocab.json")
merges_file = os.path.join(tokenizer_dir, "merges.txt")

if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
    print(f"Error: Tokenizer files not found in {tokenizer_dir}")
    print(f"Expected: {vocab_file} and {merges_file}")
else:
    print(f"Loading tokenizer from {tokenizer_dir}...")
    # PreTrainedTokenizerFast can load from the directory if it contains vocab.json and merges.txt
    # Ensure that the special tokens are correctly mapped
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    print("Tokenizer loaded successfully.")

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"])

print("Tokenization function `tokenize_function` defined.")

# Tokenize the entire dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4, # Use multiple processes for faster tokenization
    remove_columns=dataset["train"].column_names,
)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4, # Use multiple processes for faster processing
)
model_config = Config()
print(model_config)

model = DecoderOnlyTransformer(model_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model instantiated and moved to {device}.")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
print("Optimizer (AdamW) initialized with learning rate 5e-5.")

# --- From cell_id: 2913d2b5 ---
from torch.utils.data import DataLoader

# Define loss function
# CrossEntropyLoss expects logits and target indices.
# It also applies softmax internally, so don't apply softmax in the model's output.
loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id)
print("Loss function (CrossEntropyLoss) defined.")

# Create DataLoaders
# Using a small batch size for demonstration and to fit into memory.
# In a real scenario, you would tune this batch size.
batch_size = 8

train_dataloader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
)

val_dataloader = DataLoader(
    lm_datasets["validation"],
    shuffle=False,
    batch_size=batch_size,
)

test_dataloader = DataLoader(
    lm_datasets["test"],
    shuffle=False,
    batch_size=batch_size,
)

print(f"DataLoaders created with batch size: {batch_size}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of validation batches: {len(val_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")

num_epochs = 1 # Number of training epochs
from tqdm.auto import tqdm
print(f"Starting training for {num_epochs} epochs...")

for epoch in tqdm(range(num_epochs)):
    train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_dataloader, loss_fn, device)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

print("Training complete.")

prompt = "The quick brown fox"
generated_text = generate(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50,device=device)

print("Generated Text:")
print(generated_text)