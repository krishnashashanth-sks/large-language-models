from transformers import AutoTokenizer
from torch.utils.data import  DataLoader
from dataset import TextDataset
from model import GemmaLikeModel
from train import train
from inference import generate_text
from evaluate import evaluate_model
import torch.optim as optim
import torch
import torch.nn as nn

# 1. Initialize a tokenizer (using a small, fast tokenizer for demonstration)
# In a real scenario, you would choose a tokenizer appropriate for your pre-training data.
# For Gemma-like models, SentencePiece is commonly used.
# Here, we'll use a simple BERT tokenizer for demonstration.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Prepare some dummy text data
sample_texts = [
    "Hello, this is a sample sentence for tokenization.",
    "Transformers are powerful models for natural language processing.",
    "Data preparation is a crucial step in machine learning pipelines."
]

MODEL_MAX_SEQ_LEN = 128

tokenized_data = tokenizer(sample_texts,
                           padding="max_length",
                           truncation=True,
                           max_length=MODEL_MAX_SEQ_LEN,
                           return_tensors="pt") 


# Create a dataset instance
dataset = TextDataset(tokenized_data)

# Create a DataLoader instance
batch_size = 2 # For demonstration
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 1. Define learning rate
learning_rate = 5e-5

# 2. Define number of epochs
num_epochs = 3

# 3. Define batch size
batch_size = 8

# 4. Set up the computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define model parameters
embed_dim = 256
num_heads = 4
hidden_dim = 1024 # Typically 4 times embed_dim
num_layers = 2
dropout_rate = 0.1 # Already established

vocab_size = tokenizer.vocab_size # Get vocab_size from the initialized tokenizer

# 2. Instantiate the GemmaLikeModel
model = GemmaLikeModel(
    vocab_size=vocab_size,
    max_seq_len=MODEL_MAX_SEQ_LEN,
    embed_dim=embed_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout_rate=dropout_rate # Note: original model definition has dropout_date instead of dropout_rate
)

model.to(device)

# 4. Define the loss function
loss_fn = nn.CrossEntropyLoss()

# 5. Instantiate the torch.optim.AdamW optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.train()

train(num_epochs,dataloader,model,optimizer,loss_fn,device)

model.eval()

print("Running evaluation:")
eval_loss = evaluate_model(model, dataloader, loss_fn, device, vocab_size)
print(f"Final Evaluation Loss: {eval_loss:.4f}")


print("Text generation function 'generate_text' defined successfully.")

# 8. Call the generate_text function with a sample prompt and print the output
print("\n--- Demonstrating Text Generation ---")

sample_prompt = "Transformers are powerful models for"
max_gen_tokens = 50 # Number of new tokens to generate

# Assuming tokenizer has an eos_token property, or you can manually set it if known
eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
if eos_token_id is None and tokenizer.sep_token_id is not None:
    # Fallback for models like BERT where SEP token might act as an end indicator for a single sequence
    eos_token_id = tokenizer.sep_token_id

# Pass MODEL_MAX_SEQ_LEN from earlier definition

generated_output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=sample_prompt,
    max_new_tokens=max_gen_tokens,
    device=device,
    eos_token_id=eos_token_id,
    model_max_seq_len=MODEL_MAX_SEQ_LEN
)

print(f"Prompt: {sample_prompt}")
print(f"Generated Text: {generated_output}")